import telebot
from model import Model

messages = 0
positive = 0
negative = 0
concentration = 0
crit_concentration = 0.5

bot = telebot.TeleBot('1685365874:AAGKg6p9yHDGM176_5_35rV-Ai0BNFj8DRU')

model = Model()

a = model.predict(['солнце радость счастье добро'])
print(a)

@bot.message_handler(commands=['getcon'])
def get_cont(message):
    bot.send_message(445263969, concentration)

@bot.message_handler(content_types="text")
def handler_text(message):
    global messages, positive, negative, crit_concentration, concentration
    messages += 1
    print(message.text)
    print(model.predict([message.text]))
    if model.predict([message.text]) == 1:
        positive += 1
    else:
        negative += 1
    concentration = negative / messages

    if concentration > crit_concentration:
        bot.send_message(445263969, 'too many negative messages check the chat')


bot.polling(none_stop=True, interval=0)
