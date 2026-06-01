import speech_recognition as srec


def recognize_speech(rec, mic):
    with mic as source:
        rec.adjust_for_ambient_noise(source)
        print("Говоріть...")
        audio = rec.listen(source)

    result = {"Текст": None}

    try:
        result["Текст"] = rec.recognize_google(
            audio,
            show_all=False,
            language="uk-UA",
        )
    except srec.UnknownValueError:
        result["Текст"] = "Мову не розпізнано"
    except srec.RequestError as error:
        result["Текст"] = f"Помилка сервісу розпізнавання: {error}"

    return result


if __name__ == "__main__":
    recognizer = srec.Recognizer()
    mic = srec.Microphone(device_index=1)

    while True:
        result = recognize_speech(recognizer, mic)
        print("Ви сказали:\n{}".format(result["Текст"]))
