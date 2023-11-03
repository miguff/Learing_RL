import gymnasium as gym
import numpy as np

env = gym.make("MountainCar-v0", render_mode="human" ) #Meghívjuk azt a pályát amit tesztelni akarunk, vagy amit nézni
env.reset() #Beállítjuk alaphelyzetbe a környezetet


print(env.observation_space.high)#Ez a legmagasabb értéke az összes megfigyelésnek --> ezeket nem mindig tudjuk, lehet olyan szituáció, hogy nem tudjuk, csak keresés útján
print(env.observation_space.low)#Ez a legalacsonyabb értéke az összes megfigyelésnek
print(env.action_space.n) #Ez megadja, hogy mennyi action van


DISCRETE_OS_SIZE = [20] * len(env.observation_space.high) # Ez lesz a Q táblának a mérete, ezt lehet, hogy maga az ügynök fogja változtatni, meg a2 számnak nem feltétlenül kell megegyeznie
#A felette lévőből nem [40] lesz, hanem [20 20]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE
print(discrete_os_win_size)
print((env.observation_space.high - env.observation_space.low))
print(DISCRETE_OS_SIZE)

"""
Ez a Q value úgy fog kinézni mint egy táblázat
Jelen esetben, mivel csak 3 akció van, amiatt 3 oszlopa lesz
Annyi sora lesz amennyit megadjunk, ebben az esetben 20 --> Ez az adott kombinációkat jelöli amik kijöhetnek
Majd minden egyes lépés után belekerül egy kombinációba és abban a kombinációban (sorban) megkeresi, hogy melyik érték a legnagyobb és azt az akciót fogja választani
Itt van vizualizálva
https://www.youtube.com/watch?v=yMk_XtIEzH8&list=PLQVvvaa0QuDezJFIOU5wDdfy4e9vdnx-7

Majd a későbbiekben a rewardoknak köszönhetően a Q-function-el updateli a Q table-t

"""

"""
DISCRETE_OS_SIZE + [env.action_space.n]: Ezzel a paranccsal jelen pillanatban egy 20*20*3-as 3D-s mátrixot hozunk létre. Ugye a 2 20-as a lehetséges összes kombinációt takarja, a 3 pedig az akciók számát

"""

q_table = np.random.uniform(low=-2, high=0, size = (DISCRETE_OS_SIZE + [env.action_space.n])) #ezeket az értékeket lehet változtatni is, ez csak kicsit random, ez azért jó mert jelen esetben a reward -1, amíg el nem éri a zászlót, és akkor mindig 0 lesz a reward
print(q_table.shape)
print(q_table)

"""
env.observation_space.high = [0.6 0.07]
env.observation_space.low = [-1.2 -0.07]

Amit itt felette csináltunk, hogy azt a -1.2 és 0.6 közötti távolságot felosztuk 20 diszkrét értékre, ugyanez a másik oldalon is
"""

"""
3 akciót lehet tenni
action0 = balra megy
action1 = nem csinál semmit
action2 = jobbra megy
"""

# done = False #A változót hamisra tesszük, hgy addig menjen amíg nem igaz


# """
# Ez azért Q algoritmus mert létrehoz egy Q táblát, amiben az összes llapot benne van és mindegyikhez csak azokat fogja megnézni
# - Először ezeket random értékekkel tölti fel
# -Majd ahogy elindítjuk a random értékek mentén elkezd felfedezni és szépen update-eli a Q táblát
# """

# while not done: #Végtelen ciklus, most
#     action = 2 #Kiválasztjuk az első akciót
#     new_state, reward, done, truncated, info = env.step(action) #Lép egyet, és megnézi, hogy mi lett a következő helyzetben a visszakapott értékek
#     if done or truncated: #Ha végzet, akkor reseteli és megkapjuk az infókat
#         new_state , info = env.reset()
    
# env.close() #Bezárjuk a környezetet

