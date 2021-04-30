import gym
import numpy as np
import pygame
from PIL import Image
from pygame.locals import *

WIDTH = 288
HEIGHT = 412
flappyBird = pygame.image.load("Flappy.png")
pipe = pygame.image.load("pipe-green.png")


class Box:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.veloX = 0
        self.veloY = 0
        self.width = width
        self.height = height
        self.isDeath = False
        self.tag = "None"

    def draw(self, pygame, screen):
        pass

    def update(self):
        self.x += self.veloX
        self.y += self.veloY

    def col(self, obj2):
        if self.x < obj2.x + obj2.width and self.x + self.width > obj2.x and self.y < obj2.y + obj2.height and self.y + self.height > obj2.y:
            return True


class Player(Box):
    G = 1

    def __init__(self, x, y, width, height):
        global flappyBird
        super().__init__(x, y, width, height)
        self.time = 0
        self.reward = []
        self.history = []
        self.historyY = []
        flappyBird = pygame.transform.scale(flappyBird, (self.width, self.width))

    def draw(self, pygame, screen):
        # pygame.draw.rect(screen, (255, 0, 0), (int(self.x), int(self.y), self.width, self.height), 0)
        screen.blit(flappyBird, (self.x, self.y))

    def update(self):
        self.time += 1
        super().update()
        self.veloY += self.G
        self.veloY = max(self.veloY, -8)
        self.veloY = min(self.veloY, 10)
        # if self.y < 0:
        #     self.y = 0
        if self.y < 0 or self.y + self.height > HEIGHT:
            self.isDeath = True

    def jump(self):
        self.veloY = -9

    def do(self, X):
        type = np.random.choice(2, p=out)
        if type == 1:
            self.jump()
        return type

    def __lt__(self, other):
        return self.time < other.time

    def __gt__(self, other):
        return self.time > other.time


class Object(Box):
    def __init__(self, x, y, width, height, tag):
        global pipe
        super().__init__(x, y, width, height)
        self.veloX = -4
        self.tag = "OUT"
        self.tag2 = tag
        t = pipe.get_width() / width
        self.pipe2 = pygame.transform.scale(pipe, (self.width, int(pipe.get_height() * t)))
        if tag == "UP":
            self.pipe2 = pygame.transform.flip(self.pipe2, False, True)

    def draw(self, pygame, screen):
        # pygame.draw.rect(screen, (128, 255, 127), (int(self.x), int(self.y), self.width, self.height), 0
        screen.blit(self.pipe2, (self.x, self.y), (0, self.pipe2.get_height() - self.height, 52, self.height))

    def update(self):
        super().update()
        if self.x + self.width < 0:
            self.isDeath = True


class NextPoint(Box):
    def draw(self, pygame, screen):
        # pygame.draw.rect(screen, (255, 255, 255), (int(self.x), int(self.y), self.width, self.height), 0)
        pass


class Dummy(Box):
    def __init__(self, x, y, width, height):
        super().__init__(x, y, width, height)
        self.veloX = -4
        self.tag = "OK"
        self.tag2 = "no"

    def draw(self, pygame, screen):
        pass


class FlappyClass(gym.Env):
    def __init__(self):
        self.done = True
        self.player = Player(100, HEIGHT / 2, 25, 25)
        self.objects = []
        self.min = 1000
        self.time = 0
        self.isInited = False
        self.screen = None
        self.clock = None
        self.point = NextPoint(0, 0, 5, 5)
        self.action_space = gym.spaces.Discrete(2)
        self.reward = 0
        self.finish = False
        self.render()
        self.observation_space = gym.spaces.Box(
            low=0,
            high=3,
            shape=[80, 80, 4]
        )

    def render(self):
        if not self.isInited:
            pygame.init()  # 初期化
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))  # ウィンドウサイズの指定
            pygame.display.set_caption("GA_FLAPPY")
            self.clock = pygame.time.Clock()  # A clock object to limit the frame rate.
            self.isInited = True
        for event in pygame.event.get():  # 終了処理
            if event.type == KEYDOWN:
                self.player.jump()
            if event.type == QUIT:
                self.finish = True
                break
        if not self.finish:
            self.screen.fill((0, 0, 0,))

        for o in self.objects[:]:
            o.draw(pygame=pygame, screen=self.screen)
        # self.screen.blit(f, (self.player.x, self.player.y))
        self.player.draw(pygame=pygame, screen=self.screen)
        self.point.draw(pygame=pygame, screen=self.screen)
        pygame.display.update()

    def step(self, action):
        done = False
        pointFlag = False
        self.reward = 0.1
        if action == 1:
            self.player.jump()
        for o in self.objects[:]:
            o.update()
            if 0 < (o.x + o.width - self.player.x) and not pointFlag and o.tag2 == "UP":
                self.min = o.x + o.width - self.player.x
                self.point.x = o.x + o.width
                self.point.y = o.y + o.height + 60
                pointFlag = True
            if o.col(self.player):
                if o.tag == "OK":
                    o.isDeath = True
                    self.reward = 1
                else:
                    self.player.isDeath = True
            if o.isDeath:
                self.objects.remove(o)

        if not self.player.isDeath:
            self.player.update()
        else:
            self.reward = -1
            done = True

        if self.time % 50 == 0:
            rand = 0
            while rand == 0:
                rand = int(np.random.rand() * (HEIGHT - 150))
            # rand = 100
            self.objects.append(Object(WIDTH, 0, 40, rand, tag="UP"))
            self.objects.append(Object(WIDTH, rand + 120, 40, HEIGHT, tag="DOWN"))
            self.objects.append(Dummy(WIDTH + 25 + 15, rand, 5, 120))
        self.time += 1
        if not self.finish:
            self.render()
        self.clock.tick(30)

        return self.WriteState(), self.reward, done, {self.finish}

    def reset(self):
        self.player = Player(100, HEIGHT / 2, 25, 25)
        self.point = NextPoint(0, 0, 5, 5)
        self.objects = []
        self.time = 0
        self.min = 1000
        self.isInited = False
        self.screen = None
        self.clock = None
        self.reward = 0
        rand = int(np.random.rand() * (HEIGHT - 150))
        # self.objects.append(Object(self.player.x + 15, 0, 40, self.player.y - 45, tag="UP"))
        # self.objects.append(Object(self.player.x + 15, self.player.y + 120 - 45, 40, HEIGHT, tag="DOWN"))
        # self.objects.append(Dummy(self.player.x + 15 + 25 + 15, rand + self.player.y - 45, 5, 60))
        self.render()
        #
        # self.objects.append(Object(WIDTH - 70, 0, 25, rand, tag="UP"))
        # self.objects.append(Object(WIDTH - 70, rand + 75, 25, HEIGHT, tag="DOWN"))
        # self.objects.append(Dummy(WIDTH - 70 + 25 + 15, rand, 5, 60))

        return self.WriteState()

    # [self.player.y / 200, abs(self.player.y + self.player.width / 2 - self.point.y) / 200,
    #  (self.point.x - self.player.x) / 100]

    def WriteState(self):
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        pilImg = Image.fromarray(np.uint8(image_data))
        pilImg = pilImg.convert("L")
        pilImg = pilImg.resize((80, 80))
        ImgArray = np.asarray(pilImg) / 255.0
        # print(np.asarray(pilImg).shape)
        return ImgArray


if __name__ == "__main__":
    flappy = FlappyClass()
    flappy.reset()
    while True:

        _, _, done, f = flappy.step(0)
        if f.pop():
            break
        flappy.render()
        if done:
            flappy.reset()
