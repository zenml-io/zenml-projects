from enum import Enum


class Languages(str, Enum):
    def __str__(self):
        return str(self.value)

    ENGLISH = "en_XX"
    GERMAN = "de_DE"
    SPANISH = "es_XX"
    FRENCH = "fr_XX"
    RUSSIAN = "ru_RU"
    SLOVAK = "sk_SK"
    TURKISH = "tr_TR"
    FINNISH = "fi_FI"
    ESTONIAN = "et_EE"
    POLISH = "pl_PL"
    CROATIAN = "hr_HR"
    KOREAN = "ko_KR"


selected_languages = [Languages.GERMAN, Languages.SPANISH, Languages.FRENCH, Languages.TURKISH,
                      Languages.RUSSIAN, Languages.ESTONIAN, Languages.POLISH, Languages.FINNISH]

selected_languages_codes = [k.value for k in selected_languages]
