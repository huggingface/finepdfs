from datatrove.pipeline.base import PipelineStep
import re
import numpy as np
from datatrove.utils.media import iter_pages


class RemoveImageAnnotationsByRatio(PipelineStep):
    """Remove text between <docling_picture_annotation> and <docling_picture_annotation_non_text> tags if alpha/all_char ratio is below threshold"""
    name: str = "🖼️ RemoveImageAnnotationsByRatio"
    type: str = "🔧 Filter"
    
    def __init__(self, ratio_threshold: float):
        self.ratio_threshold = ratio_threshold
        super().__init__()
    
    def run(self, data, rank: int = 0, world_size: int = 1):
        # Compile regex for performance - handles both variants
        annotation_regex = re.compile(r"<docling_picture_annotation(?:_non_text)?>(.*?)</docling_picture_annotation(?:_non_text)?>", re.DOTALL)
        
        for document in data:
            
            def replacement_func(match):
                annotation_text = match.group(1)  # The content is in group 1 with non-capturing groups
                
                # Calculate alpha/all_char ratio
                if len(annotation_text) == 0:
                    return ""  # Remove empty annotations
                
                alpha_count = sum(c.isalpha() for c in annotation_text)
                ratio = alpha_count / len(annotation_text)
                
                # Remove if ratio is below threshold
                self.stat_update("image_annotations_processed", 1, unit="document")
                if ratio < self.ratio_threshold:
                    self.stat_update("removed_image_annotations", 1, unit="document")
                    return ""
                else:
                    # Keep the annotation with its tags
                    return match.group(0)
            
            # Apply replacement function to all matches
            with self.track_time():
                pages = []
                for page in iter_pages(document):
                    pages.append(annotation_regex.sub(replacement_func, page))
                document.text = "".join(pages)
                document.media[0].metadata["page_offsets"] = [int(offset) for offset in np.cumsum([len(page) for page in pages])]
            yield document