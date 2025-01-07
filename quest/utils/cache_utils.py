import logging
from typing import List, Optional, Tuple
import torch
import math
from transformers import DynamicCache

logger = logging.getLogger(__name__)

    
class RaaSCache(DynamicCache):

    """
    The following implementation assumes that the cache is not full at the beginning of the decoding process.
    """

    def __init__(self, page_size: int, cache_budget: int, num_hidden_layers: Optional[int] = None) -> None:
        super().__init__(num_hidden_layers)
        self.page_size = page_size # tot number of tokens per page
        self.cache_budget = cache_budget # tot number of tokens in the cache
        self.page_budget = self.cache_budget // self.page_size # tot number of pages in the cache
        self.page_id_to_access_status: List[torch.Tensor] = []
        self.counter = 0
        self.max_num_pages = 2**12 # 4096 pages with page_size 16

    def get_attention_mask(self, attn_weights: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Return the attention mask which masks outdated cache.

        Args: 
            attn_weights: torch.tensor of shape (batch_size, num_heads, q_len, seq_len)
            The attention weights of the current layer.
            layer_idx: The index of the current layer.

        Returns:
            torch.tensor of shape (batch_size, num_heads, q_len, seq_len)
            The attention mask which masks outdated cache.
        """

        # The first decode. We do not use attetion map in prefill stage and the first decode
        if len(self.page_id_to_access_status) <= layer_idx:
            return None
        # We do not use attetion map if the cache is not full
        if self._seen_tokens <= self.cache_budget:
            return None

        # import pdb; pdb.set_trace()
        bzs, num_heads, q_len, seq_len = attn_weights.shape
        attention_mask = torch.ones(bzs, num_heads, q_len, (seq_len + self.page_size - 1) // self.page_size * self.page_size, device=self.key_cache[0].device) * torch.tensor(torch.finfo(self.key_cache[0].dtype).min)
        _, topk = self.page_id_to_access_status[layer_idx].topk(self.page_budget, dim=-1)
        topk = topk.unsqueeze(-1).repeat(1, 1, 1, 1, self.page_size) * self.page_size + torch.arange(
            self.page_size, device=topk.device
        )
        topk = topk.reshape(topk.shape[0], topk.shape[1], topk.shape[2], -1)
        attention_mask.scatter_(-1, topk, 0)  
        attention_mask[..., -self.page_size:] = 0 # Novice protection for the last page
        return attention_mask[..., :seq_len]

    
    def update_access_history(self, access_page_ids, access_page_scores, layer_idx: int):
        """
        Update the access history of the cache.

        Args:
            access_page_ids: torch.tensor of shape (batch_size, num_heads, q_len, num_pages)
            The page ids that are accessed in the current step. 
            access_page_scores: torch.tensor of shape (batch_size, num_heads, q_len, num_pages)
            The scores of the accessed pages.
            layer_idx: The index of the current layer.
        
        Returns:
            None
        """
        assert access_page_ids.shape[0] == 1 and access_page_ids.shape[2] == 1, "We only support 1 batch size and 1 q for now."
        assert access_page_scores.shape[0] == 1 and access_page_scores.shape[2] == 1, "We only support 1 batch size and 1 q for now."

        # The first decode
        if len(self.page_id_to_access_status) <= layer_idx:
            # There may be skipped layers, fill them with empty lists
            # For example, Quest skips the first 2 layers
            for _ in range(len(self.page_id_to_access_status), layer_idx+1):
                self.page_id_to_access_status.append(torch.zeros(
                    access_page_ids.shape[0], # batch_size 1
                    access_page_ids.shape[1], # num_heads
                    access_page_ids.shape[2], # q_len 1
                    self.max_num_pages, # num_pages
                    device=access_page_ids.device
                ))
            

        self.counter = self._seen_tokens # Motonically increasing counter
        # Only the top-(k/2) is deemed as accessed as important pages
        self.page_id_to_access_status[layer_idx].scatter_(-1, access_page_ids[..., :self.page_budget // 2], self.counter)



class H2OCache(DynamicCache):
    """
    The following implementation assumes that the cache is not full at the beginning of the decoding process.
    """

    def __init__(self, cache_budget: int, num_hidden_layers: Optional[int] = None) -> None:
        super().__init__(num_hidden_layers)
        self.cache_budget = cache_budget # tot number of tokens in the cache
        self.token_id_to_cum_attn_score: List[torch.Tensor] = []
        self.max_num_tokens = 2**16 # 32k tokens

    def get_attention_mask(self, attn_weights: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Return the attention mask which masks outdated cache.

        Args: 
            attn_weights: torch.tensor of shape (batch_size, num_heads, q_len, seq_len)
            The attention weights of the current layer.
            layer_idx: The index of the current layer.

        Returns:
            torch.tensor of shape (batch_size, num_heads, q_len, seq_len)
            The attention mask which masks outdated cache.
        """

        # The first decode. We do not use attetion map in prefill stage and the first decode
        if len(self.token_id_to_cum_attn_score) <= layer_idx:
            return None
        # We do not use attetion map if the cache is not full
        if self._seen_tokens <= self.cache_budget:
            return None

        
        attention_mask = torch.ones_like(attn_weights, device=self.key_cache[0].device) * torch.tensor(torch.finfo(self.key_cache[0].dtype).min)
        _, topk = self.token_id_to_cum_attn_score[layer_idx][..., :attn_weights.shape[-1]-self.cache_budget//2].topk(
            k=self.cache_budget // 2, dim=-1
        )
        # import pdb; pdb.set_trace()


        mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
        mask_bottom.scatter_(-1, topk, True)
        mask_bottom[..., -self.cache_budget // 2 :] = True

        assert (mask_bottom.sum(dim=-1) == self.cache_budget).all(), "The number of tokens to discard should be equal to cache_budget"

        # Zero out the accumulated attention weights of the discarded tokens
        self.token_id_to_cum_attn_score[layer_idx][..., :attention_mask.shape[-1]] *= mask_bottom

        return attention_mask

    
    def update_access_history(self, attn_weights: torch.Tensor, layer_idx: int):
        """
        Update the access history of the cache.

        Args:
            Args: 
            attn_weights: torch.tensor of shape (batch_size, num_heads, q_len, seq_len)
            The attention weights of the current layer.
            layer_idx: The index of the current layer.
        
        Returns:
            None
        """

        # The first decode
        if len(self.token_id_to_cum_attn_score) <= layer_idx:
            # There may be skipped layers, fill them with empty lists
            # For example, Quest skips the first 2 layers
            for _ in range(len(self.token_id_to_cum_attn_score), layer_idx+1):
                self.token_id_to_cum_attn_score.append(torch.zeros(
                    attn_weights.shape[0], # batch_size 1
                    attn_weights.shape[1], # num_heads
                    attn_weights.shape[2], # q_len 1
                    self.max_num_tokens,
                    device=attn_weights.device
                ))
            
        self.token_id_to_cum_attn_score[layer_idx][..., :attn_weights.shape[-1]] += attn_weights