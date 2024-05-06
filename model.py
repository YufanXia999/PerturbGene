"""
Essentially an adaptation of BERT where the positional embeddings are removed.
However, this is implemented by overriding DistilBERT classes because only DistilBERT supports Flash Attention.
Hence, there is significantly more code because vanilla DistilBERT has no `token_type_ids`.
"""

from typing import Optional, Tuple, Union

import torch
from torch import nn
import transformers
from transformers.activations import get_activation
from transformers.modeling_outputs import MaskedLMOutput, BaseModelOutput, SequenceClassifierOutput
from transformers.utils import logging


logger = logging.get_logger(__name__)


class PositionlessEmbeddings(nn.Module):
    """
    Copied from `BertEmbeddings`, not DistilBERT embeddings because DistilBERT does not use `token_type_ids`.
    Differences from `BertEmbeddings` are:
        - `position_embeddings` are removed.
        - registered_buffers are removed because we assume `token_type_embeddings` are always passed in

    An expressed gene is the sum of:
        the "word_embedding", which represents the expression level (i.e. bin number)
        the "token_type_embedding", which represents the specific gene

    Therefore, in our case we will likely have significantly more distinct token_types (genes) than words (bins).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        assert token_type_ids is not None, "Expecting token_type_ids to always be specified for genes."

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class GeneBertModel(transformers.DistilBertPreTrainedModel):
    """
    Equivalent to `DistilBertModel` but with `PositionlessEmbeddings`
    and allowing `token_type_ids` argument in `forward()`.
    """
    def __init__(self, config: transformers.PretrainedConfig):
        """
        Only modification is using `PositionlessEmbeddings`.
        """
        super().__init__(config)

        self.embeddings = PositionlessEmbeddings(config)  # changed to `PositionlessEmbeddings`
        self.transformer = transformers.models.distilbert.modeling_distilbert.Transformer(config)  # encoder
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings: nn.Embedding):
        self.embeddings.word_embeddings = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        token_type_ids: Optional[torch.Tensor] = None,  # added - KZ
    ) -> Union[BaseModelOutput, Tuple[torch.Tensor, ...]]:
        assert token_type_ids is not None, "`token_type_ids` are optional for backwards-compatible argument order, " \
            "but they should always be specified for genes."  # added - KZ
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        embeddings = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,  # added - Kevin
            inputs_embeds=inputs_embeds
        )  # (bs, seq_length, dim)
        if self._use_flash_attention_2:
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            if attention_mask is None:
                attention_mask = torch.ones(input_shape, device=device)  # (bs, seq_length)

        return self.transformer(
            x=embeddings,
            attn_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class GeneBertForPhenotypicMLM(transformers.DistilBertPreTrainedModel):
    """
    Copied from `DistilBertForMaskedLM`. Changes are:
        - Changed `self.distilbert` from `DistilBertModel` to `GeneBertModel`.
        - Added `token_type_ids` to `forward()` and propogating it to `self.distilbert`.
        - Changed `mlm_loss` to use `label_smoothing`.
        - Omitted `get_position_embeddings()` and `resize_position_embeddings()` since `GeneBertModel` ignores position.
    """

    _tied_weights_keys = ["vocab_projector.weight"]

    def __init__(self, config: transformers.PretrainedConfig, mlm_loss_fct: nn.CrossEntropyLoss | None = None):
        super().__init__(config)

        self.activation = get_activation(config.activation)

        self.distilbert = GeneBertModel(config)  # changed to `GeneBertModel` - Kevin
        self.vocab_transform = nn.Linear(config.dim, config.dim)
        self.vocab_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)
        self.vocab_projector = nn.Linear(config.dim, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

        if mlm_loss_fct is not None:
            self.mlm_loss_fct = mlm_loss_fct
        else:
            self.mlm_loss_fct = nn.CrossEntropyLoss(label_smoothing=0.5)  # added `label_smoothing` - Kevin

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        token_type_ids: Optional[torch.Tensor] = None,  # added - Kevin
    ) -> Union[MaskedLMOutput, Tuple[torch.Tensor, ...]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        dlbrt_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            token_type_ids=token_type_ids,  # added - Kevin
        )
        hidden_states = dlbrt_output[0]  # (bs, seq_length, dim)
        prediction_logits = self.vocab_transform(hidden_states)  # (bs, seq_length, dim)
        prediction_logits = self.activation(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)

        mlm_loss = None
        if labels is not None:
            mlm_loss = self.mlm_loss_fct(prediction_logits.view(-1, prediction_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (prediction_logits,) + dlbrt_output[1:]
            return ((mlm_loss,) + output) if mlm_loss is not None else output

        return MaskedLMOutput(
            loss=mlm_loss,
            logits=prediction_logits,
            hidden_states=dlbrt_output.hidden_states,
            attentions=dlbrt_output.attentions,
        )


class GeneBertForClassification(transformers.DistilBertPreTrainedModel):
    """
    Copied from `DistilBertForSequenceClassification`. Changes are:
        - Changed `self.distilbert` from `DistilBertModel` to `GeneBertModel`.
        - Added `token_type_ids` to `forward()` and propogating it to `self.distilbert`.
        - Omitted `get_position_embeddings` and `resize_position_embeddings` since `GeneBertModel` ignores position.
        - Added `cls_loss_fct` to __init__() and use it in `forward()`.
    """
    def __init__(self, config: transformers.PretrainedConfig, cls_loss_fct: nn.CrossEntropyLoss | None = None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.distilbert = GeneBertModel(config)  # changed to `GeneBertModel` - Kevin
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        self.cls_loss_fct = cls_loss_fct  # added - Kevin

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        token_type_ids: Optional[torch.Tensor] = None,  # added - Kevin
    ) -> Union[SequenceClassifierOutput, Tuple[torch.Tensor, ...]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            token_type_ids=token_type_ids,  # added - Kevin
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)

        loss = None
        if labels is not None:
            if self.cls_loss_fct is not None:  # added if/else - Kevin
                # assuming "single_label_classification" with > 1 label for now
                loss = self.cls_loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                if self.config.problem_type is None:
                    if self.num_labels == 1:
                        self.config.problem_type = "regression"
                    elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                        self.config.problem_type = "single_label_classification"
                    else:
                        self.config.problem_type = "multi_label_classification"

                if self.config.problem_type == "regression":
                    loss_fct = nn.MSELoss()
                    if self.num_labels == 1:
                        loss = loss_fct(logits.squeeze(), labels.squeeze())
                    else:
                        loss = loss_fct(logits, labels)
                elif self.config.problem_type == "single_label_classification":
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                elif self.config.problem_type == "multi_label_classification":
                    loss_fct = nn.BCEWithLogitsLoss()
                    loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + distilbert_output[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )
