import os

from dotenv import load_dotenv

from . import read_one_img_to_matrix, neuron_net


def recognition(img_path):
    img_matrix = read_one_img_to_matrix(img_path)

    load_dotenv()
    scales_index = int(os.getenv('scales_index'))
    layer_matrices = img_matrix.iloc[0]

    recognizer = neuron_net(layer_matrices, scales_index)[0]
    get_answer = recognizer.idxmax()
    services = {
        0: "WhatsApp",
        1: "Facebook",
        2: "Skype",
        3: "Viber",
        4: "Kik",
        5: "Vk",
        6: "Telegram",
        7: "Discord",
        8: "Snapchat",
        9: "ICQ"
    }

    return services.get(get_answer, "Invalid value. Please enter a number from 0 to 9.")
