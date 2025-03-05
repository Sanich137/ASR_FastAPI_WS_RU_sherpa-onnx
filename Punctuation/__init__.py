from utils.pre_start_init import paths
from utils.do_logging import logger
from .punctuate import SbertPuncCaseOnnx

try:
    sbertpunc = SbertPuncCaseOnnx(paths.get("punctuation_model_path"))
except Exception as e:
    logger.error(f"Error gatting punctuation model - {e}")
else:
    input_text = "channel_1: два\nchannel_2: татьяна добрый день это компания сберправа вас беспокоит меня зовут ульяна\nchannel_2: звоню уточнить по поводу документов мы у вас в чате запрашивали список документов\nchannel_2: скажите пожалуйста когда сможете прислать чтобы юристы ознакомились с ними\nchannel_1: два сегодня а да ну хорошо а доброго\nchannel_2: сегодня пришлете хорошо тогда ждем от вас всего доброго\nchannel_2: до свидания\n"
    print(f"Source text:   {input_text}\n")
    punctuated_text = sbertpunc.punctuate(input_text)
    print(f"Restored text: {punctuated_text}")
