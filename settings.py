import locale
import os
import platform


# 경로 설정
# directory = "C:/PycharmProjects/RL_Trader/rltrader-master/"
# BASE_DIR = os.path.dirname(os.path.abspath(directory))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))



# 로케일 설정
if 'Linux' in platform.system() or 'Darwin' in platform.system():
    try:
      locale.setlocale(locale.LC_ALL, 'ko_KR.UTF-8')
    except:
      None
elif 'Windows' in platform.system():
    locale.setlocale(locale.LC_ALL, '')
print(BASE_DIR)
