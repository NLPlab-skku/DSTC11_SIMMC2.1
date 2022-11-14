from utils.mask import BertMasker
from transformers import BartForConditionalGeneration as BartLMHeadModel
from torch import nn
from torch.nn import CrossEntropyLoss
import torch
from transformers import PreTrainedModel, BartPretrainedModel, BartModel

from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

def pp_batch(input_ids,input_mask,labels,label_mask):
    
    # first
    batch_size = input_ids.shape[0]
    cls_input = torch.cat((input_ids,labels),1)
    cls_labels = torch.tensor([0]*batch_size)
    
    # second
    cls_input = torch.cat((cls_input,torch.cat((input_ids,labels.flip(0)),1)),0)
    cls_labels = torch.cat((cls_labels,torch.tensor([1]*batch_size)),0)
                           
    #third (if needed)
    
    # attention mask
    cls_attention_mask = torch.cat((input_mask,label_mask),1)
    flipped_attention_mask = torch.cat((input_mask,label_mask.flip(0)),1)
    cls_attention_mask = torch.cat((cls_attention_mask,flipped_attention_mask),0)
    
    return cls_input, cls_attention_mask,cls_labels
    

# masking은 어디서 하게 되는거지?
class BartForMaskedLM(BartPretrainedModel):
    
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head.weight"]
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = BartModel(config)

        self.act_fun = nn.Softmax(dim=-1)
        self.cls_norm = nn.LayerNorm(config.hidden_size, eps=1e-08)        
       
        self.loss_fct = CrossEntropyLoss()
        
        self.cls_pooler = nn.Linear(config.hidden_size,config.hidden_size)
        self.cls_classifier = nn.Linear(config.hidden_size,2)       
                           
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        
        # Initialize weights and apply final processing
        self.post_init()

    def set_masker(self, tokenizer):
        self.masker=BertMasker(tokenizer)
        
    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()
    
    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        self.resized_vocab_size = new_num_tokens
        return new_embeddings    

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings    
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        label_attention = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pretrain = False,
        use_mlm = False,
        use_bart_mlm = False,
        use_encoder_mlm = False,
        use_cls = False,
        use_eos = False,
        
        pos_input_ids: torch.LongTensor = None,
        pos_attention_mask: Optional[torch.Tensor] = None,
        pos_labels: Optional[torch.LongTensor] = None,
        pos_label_attention = None,
        
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        """
        
        return_dict = return_dict # if return_dict is not None else self.config.use_return_dict # ...????
        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        
        """
            for pretraining 
        """
        lamb = 1

        bart_masked_lm_loss = None
        encoder_masked_lm_loss = None
        #  BART 의 pretraining 기법을 따라해서 LM loss 를 구해주게됨. / encoder mlm도 설정 가능
        if use_mlm:
            device = input_ids.device
            pretrain_labels = torch.cat((input_ids,labels), 1)
            
            concat_input_ids = torch.cat((pos_input_ids,pos_labels), 1)
            mlm_attention_mask = torch.cat((pos_attention_mask,pos_label_attention),1)

            # random masking part
            masked_input_ids, masked_encoder_labels= self.masker.mask_tokens(concat_input_ids.to("cpu"),0.15)    
            masked_input_ids = masked_input_ids.to(device)
            masked_encoder_labels = masked_encoder_labels.to(device)

            pretrain_decoder_input_ids = shift_tokens_right(
                pretrain_labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )
            
            masked_outputs = self.model(
                masked_input_ids,
                attention_mask=mlm_attention_mask,
                decoder_input_ids=pretrain_decoder_input_ids,
                encoder_outputs=encoder_outputs,
                decoder_attention_mask=decoder_attention_mask,
                head_mask=head_mask,
                decoder_head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            masked_logits = self.lm_head(masked_outputs["last_hidden_state"]) + self.final_logits_bias
            encoder_mlm_logits = self.lm_head(masked_outputs["encoder_last_hidden_state"]) + self.final_logits_bias
            
            loss_fct = CrossEntropyLoss()

            if use_bart_mlm is not None:

                bart_masked_lm_loss = loss_fct(masked_logits.view(-1, self.resized_vocab_size), pretrain_labels.view(-1))
                bart_masked_lm_loss = bart_masked_lm_loss * lamb
                
            if use_encoder_mlm is not None:
                encoder_masked_lm_loss = loss_fct(encoder_mlm_logits.view(-1, self.resized_vocab_size), masked_encoder_labels.view(-1))
                encoder_masked_lm_loss = encoder_masked_lm_loss * lamb                
            
        # CLS를 이용한 pretraining task 구성
        cls_loss = None
        if use_cls: #  training 일 시에만 MLM loss 구함
            device = input_ids.device
            cls_input_ids, cls_attention_mask, cls_labels = pp_batch(input_ids, attention_mask, labels, label_attention)
            cls_labels = cls_labels.to(device)
            
            encoder_hidden_states = self.model.encoder(
                    input_ids=cls_input_ids,
                    attention_mask=cls_attention_mask,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )["last_hidden_state"]
            
            prediction_scores = self.cls_classifier(self.cls_norm(self.act_fun(self.cls_pooler(encoder_hidden_states[:,0]))))
            cls_loss = self.loss_fct(prediction_scores.view(-1, 2), cls_labels.view(-1)) # 펴주는 과정
            cls_loss = cls_loss*lamb
        
        pretrain_loss = torch.tensor(0).float()
        if pretrain:
            device = input_ids.device
            pretrain_loss = pretrain_loss.to(device)
            if use_cls:
                pretrain_loss += cls_loss
            
            if use_bart_mlm:
                pretrain_loss += bart_masked_lm_loss
            
            if use_encoder_mlm:
                pretrain_loss += encoder_masked_lm_loss
            
            return pretrain_loss              
        
        """
            original loss
        """          

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.resized_vocab_size), labels.view(-1))
            if use_mlm:
                masked_lm_loss = masked_lm_loss + encoder_masked_lm_loss
   
        if not return_dict:
            output = (lm_logits,) + outputs[1:]

            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past
    
    