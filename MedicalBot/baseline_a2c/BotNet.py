import argparse
from tianshou.env import SubprocVectorEnv
import pickle
from MaskEnvrionment import MedicalEnvrionment
from Utils import *
import torch
from NeuralNet import NeuralNet
import json
import random
import nltk
from NLP import tokenize, stem, bag_of_words
nltk.download('punkt')



class BotNet:

    def __init__(self):
        self.bot_name = "Sam"
        self.disease = ""
        return

    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', type=int, default=1626)
        parser.add_argument('--buffer-size', type=int, default=20480)
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--gamma', type=float, default=0.99)
        parser.add_argument('--epoch', type=int, default=100)
        parser.add_argument('--step-per-epoch', type=int, default=256)
        parser.add_argument('--collect-per-step', type=int, default=512)
        parser.add_argument('--repeat-per-collect', type=int, default=1)
        parser.add_argument('--batch-size', type=int, default=128)
        parser.add_argument('--layer-num', type=int, default=2)
        parser.add_argument('--training-num', type=int, default=10)
        parser.add_argument('--test-num', type=int, default=1)
        parser.add_argument('--logdir', type=str, default='log')
        parser.add_argument('--render', type=float, default=0.)

        parser.add_argument(
            '--device', type=str,
            default='cuda' if torch.cuda.is_available() else 'cpu')
        # a2c special
        parser.add_argument('--vf-coef', type=float, default=0.01)
        parser.add_argument('--ent-coef', type=float, default=0.75)
        parser.add_argument('--max-grad-norm', type=float, default=None)
        parser.add_argument('--max_episode_steps', type=int, default=22)
        parser.add_argument('--logpath', type=str, default='a2c/')
        return parser.parse_args("")

    def reload(self, userInput):
        args = self.get_args()
        slot_set = []

        with open('./dataset/slot_set.txt', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                slot_set.append(line.strip())
        # slot_set =
        goals = {}
        with open('./dataset/test_goals.pk', 'rb') as f:
            goals['test'] = pickle.load(f)
        f.close()
        print(goals)

        total_disease = []
        with open('./dataset/disease.txt', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                total_disease.append(line.strip())
        print(len(slot_set), slot_set)
        disease_num = len(total_disease)

        env = MedicalEnvrionment(slot_set, goals['test'], disease_num=disease_num)
        args.state_shape = env.observation_space.shape or env.observation_space.n
        args.action_shape = env.action_space.shape or env.action_space.n

        userInput = [userInput]

        environment = MedicalEnvrionment(slot_set, userInput, max_turn=args.max_episode_steps, flag="test",
                                        disease_num=disease_num)
        test_envs = SubprocVectorEnv(
            [lambda: environment
             for _ in range(args.test_num)])

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        test_envs.seed(args.seed)
        random.seed(args.seed)
        FILE = 'model/ehr/policy.pth'
        policy = torch.load(FILE)  # Carica il modello
        test_collector = MyCollector(policy, test_envs)
        result = test_episode(policy, test_collector, test_fn=None, epoch=1,
                              n_episode=len(userInput), writer=None)
        self.disease = result['disease']
        return result

    def search_symptom(self, sentence):
        userSymptom = {
            'explicit_inform_slots': {},
            'implicit_inform_slots': {},
            'disease_tag': ""
        }
        symptoms = np.empty((0, 118))
        with open('dataset/symptom.txt') as f:
            lines = f.readlines()
            for el in lines:
                el = el.replace('\n', '')
                symptoms = np.append(symptoms, el)

        phrase = sentence.lower()
        for s in symptoms:
            if phrase.find(s.lower()) >= 0:
                userSymptom['explicit_inform_slots'].update({s: True})
        return userSymptom

    def get_response(self, sentenceFull):
        # Implementazione della chat
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with open("dataset/IntentsIlls.json", "r") as f:
            intents = json.load(f)

        FILE = "model/data.pth"
        data = torch.load(FILE)

        input_size = data["input_size"]
        hidden_size = data["hidden_size"]
        output_size = data["output_size"]
        all_words = data["all_words"]
        tags = data["tags"]
        model_state = data["model_state"]

        model = NeuralNet(input_size, hidden_size, output_size).to(device)
        model.load_state_dict(model_state)
        model.eval()

        sentence = tokenize(sentenceFull)

        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X)

        output = model(X)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            for intent in intents["intents"]:
                if tag == intent["tag"]:
                    result = random.choice(intent['responses'])
                    if tag == "symptoms":
                        userSymptom = self.search_symptom(sentence=sentenceFull)
                        result = self.reload(userInput=userSymptom)
                        print(result)
                        return f"I think that you have this disease: {self.disease}"
                    return result
        else:
            result = "I do not understand..."
            return result
