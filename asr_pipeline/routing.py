from .pipeline_config import PipelineStage
from models.fast_api_models import PostFileRequest
from utils.do_logging import logger

class PipelineRouter:
    def __init__(self, config: list[PipelineStage]):
        self.config_map = {stage.name: stage for stage in config}

    def get_next_stage(self, current_stage: str, params: PostFileRequest) -> str:
        stage = self.config_map[current_stage]
        logger.info(f"для определения следующего этапа получено stage.name: {stage.name}")

        # Todo можно попробовать внести в PIPELINE_CONFIG отсылку на params и if/else сократится до одного условия
        next_stage = self.config_map[stage.name].next_stage[0]

        if current_stage == "asr":
            if not params.do_echo_clearing:
                self.get_next_stage(next_stage, params=params)
        elif current_stage == "echo_clearing":
            if not params.do_diarization:
                self.get_next_stage(next_stage, params=params)
        elif current_stage == "diarize":
            if not params.do_dialogue:
                self.get_next_stage(next_stage, params=params)
        elif current_stage == "dialogue":
            if not params.do_punctuation:
                self.get_next_stage(next_stage, params=params)

        return next_stage