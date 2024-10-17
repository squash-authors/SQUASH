from qsession import QSession
from typing import TypedDict, Tuple


class TransformationSummary(TypedDict, total=False):
    """ State that is passed between pieplined transformation elements """
    instantiated: Tuple[str, ...]
    created: Tuple[str, ...]
    modified: Tuple[str, ...]
    removed: Tuple[str, ...]
    processed: Tuple[str, ...]    
    extra: any


class PipelineElement:
    def __init__(self, session: QSession):
        self.session = session

    def process(self, pipe_state: TransformationSummary = None) -> TransformationSummary:
        raise NotImplementedError("Child classes should implement this method")
