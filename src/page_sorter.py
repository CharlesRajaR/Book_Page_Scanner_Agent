import logging
import re
from typing import List, Optional

from src.models import ProcessedPageItem

logger = logging.getLogger(__name__)


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()


def _sentence_continuation_score(prev_last: str, next_first: str) -> float:
    """
    Heuristic linguistic continuity score between the end of Page A and start of Page B.
    - Higher score if Page A ends mid-sentence (no terminal punctuation) and Page B starts lowercase.
    - Bonus if lexical overlap exists across hyphenated word breaks.
    """
    if not prev_last or not next_first:
        return 0.0

    score = 0.0
    prev_stripped = prev_last.strip()
    next_stripped = next_first.strip()

    # Check if previous page ends mid-sentence
    if prev_stripped and prev_stripped[-1] not in '.!?':
        score += 0.5
        # Check if next page begins with a lowercase continuation word
        first_alpha = next((ch for ch in next_stripped if ch.isalpha()), "")
        if first_alpha and first_alpha.islower():
            score += 0.4

    # Hyphenated word wrap across page boundary
    if prev_stripped.endswith("-"):
        score += 0.3

    return score


def order_pages(pages: List[ProcessedPageItem]) -> List[ProcessedPageItem]:
    """
    Order processed single-page items into their true book reading order:
    1. Partition pages into `numbered` (explicit printed page number) and `unnumbered`.
    2. Sort `numbered` pages ascending by printed page number (tie-breaking by original capture order).
    3. Slot `unnumbered` pages into gaps:
       - If an unnumbered page sits between two consecutive numbered pages in original capture order
         or matches sentence continuity after a numbered page, insert it right after that predecessor.
       - Otherwise, place front-matter (covers/title pages) before the first numbered page,
         and trailing unnumbered pages at the end.
    """
    if not pages:
        return []

    numbered: List[ProcessedPageItem] = []
    unnumbered: List[ProcessedPageItem] = []

    for item in pages:
        nums = item.analysis.page_numbers if item.analysis else []
        if nums:
            # Primary sort key is the printed page number; secondary is capture sequence
            item.assigned_sort_key = float(nums[0]) + (item.original_index * 1e-4)
            numbered.append(item)
        else:
            unnumbered.append(item)

    numbered.sort(key=lambda x: x.assigned_sort_key)

    if not numbered:
        # Fallback if NO pages had printed numbers: preserve capture order + side ('left' before 'right')
        logger.warning("No printed page numbers detected on any page; falling back to capture order.")
        return sorted(
            unnumbered,
            key=lambda x: (x.original_index, 0 if x.sub_page_side == "left" else 1),
        )

    if not unnumbered:
        return numbered

    # Resolve unnumbered pages
    ordered = list(numbered)
    for un_item in unnumbered:
        best_insert_idx: Optional[int] = None
        best_score = 0.0

        un_first = un_item.analysis.first_sentence_snippet if un_item.analysis else ""
        un_last = un_item.analysis.last_sentence_snippet if un_item.analysis else ""
        is_cover = un_item.analysis.is_blank_or_cover if un_item.analysis else False

        # Front-matter / cover heuristic: captured before first numbered page and flagged as cover/blank
        first_numbered = ordered[0]
        if is_cover and un_item.original_index < first_numbered.original_index:
            ordered.insert(0, un_item)
            logger.info(
                "Placed unnumbered front-matter '%s' before page %s",
                un_item.processed_image.name,
                first_numbered.analysis.page_numbers[0],
            )
            continue

        # Check sentence continuity against existing ordered pages
        for idx, candidate in enumerate(ordered):
            cand_last = candidate.analysis.last_sentence_snippet if candidate.analysis else ""
            score = _sentence_continuation_score(cand_last, un_first)

            # Bonus if captured immediately after candidate in original sequence
            if un_item.original_index == candidate.original_index + 1:
                score += 0.25

            if score > best_score:
                best_score = score
                best_insert_idx = idx + 1

        if best_insert_idx is not None and best_score >= 0.5:
            ordered.insert(best_insert_idx, un_item)
            logger.info(
                "Slotted unnumbered page '%s' at index %d (continuity score=%.2f)",
                un_item.processed_image.name,
                best_insert_idx,
                best_score,
            )
        else:
            # Default: place relative to original capture index
            inserted = False
            for idx, candidate in enumerate(ordered):
                if un_item.original_index < candidate.original_index:
                    ordered.insert(idx, un_item)
                    inserted = True
                    break
            if not inserted:
                ordered.append(un_item)

    return ordered
