import logging
from typing import List, Optional, Tuple
import torch

from transformers import DynamicCache

logger = logging.getLogger(__name__)

    
class RaaSCache(DynamicCache):

    def __init__(self, page_size: int, cache_budget: int, num_hidden_layers: Optional[int] = None) -> None:
        super().__init__(num_hidden_layers)
        self.page_size = page_size
        self.cache_budget = cache_budget
        self.page_id_to_access_status = None
        self.counter = 0
        self.max_num_pages = 2**16

    def get_attention_mask(self, attention_mask_shape: Tuple[int, int, int, int]) -> torch.Tensor:
        """
        Return the attention mask which masks outdated cache.

        Args: 
            attention_mask_shape (:obj:`Tuple[int, int, int, int]`): The shape of the attention mask tensor.
            bzs, num_heads, q_len, seq_len = attention_mask_shape
        """
        # Before prefill, we do not mask any page
        if self.counter == 0:
            return None

        bzs, num_heads, q_len, seq_len = attention_mask_shape
        attention_mask = torch.zeros(bzs, num_heads, q_len, seq_len, device=self.key_cache[0].device)
        _, topk = self.page_id_to_access_status.topk(self.cache_budget // self.page_size, dim=-1)
        topk = topk.unsqueeze(-1).repeat(1, 1, 1, 1, self.page_size) * self.page_size + torch.arange(
            self.page_size, device=topk.device
        )
        topk = topk.reshape(topk.shape[0], topk.shape[1], topk.shape[2], -1)
        attention_mask[topk] = torch.tensor(torch.finfo(self.key_cache[0].dtype).min)

    
    def update_access_history(self, access_page_ids, access_page_scores):
        """
        Update the access history of the cache.

        Args:
            access_page_ids: torch.tensor of shape (batch_size, num_heads, q_len, num_pages)
            The page ids that are accessed in the current step. 
            access_page_scores: torch.tensor of shape (batch_size, num_heads, q_len, num_pages)
            The scores of the accessed pages.

        """
        assert access_page_ids.shape[0] == 1 and access_page_ids.shape[2] == 1, "We only support 1 batch size and 1 q for now."
        assert access_page_scores.shape[0] == 1 and access_page_scores.shape[2] == 1, "We only support 1 batch size and 1 q for now."

        if self.counter == 0:
            # (num_heads, num_pages)
            self.page_id_to_access_status = torch.zeros(
                access_page_ids.shape[0], # batch_size 1
                access_page_ids.shape[1], # num_heads
                access_page_ids.shape[2], # q_len 1
                self.max_num_pages, # num_pages
                device=access_page_ids.device
            )
            

        self.counter += 1
        # Only the top-k/2 is deemed as accessed as important pages
        self.page_id_to_access_status[access_page_ids[..., :access_page_ids.shape[-1] // 2]] = self.counter



        