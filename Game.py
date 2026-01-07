# Game.py
import random
import pygame


class Game:
    """
    Dino Runner RL Environment (vereinfacht) mit pygame.
    Actions: 0 = nichts, 1 = springen

    Verbesserungen:
    - Viel feinerer State (dist_bin + y_bin + vy_bin) -> Timing lernbar
    - "Easy training" möglich (fixe Hindernisgrössen am Anfang) -> Curriculum
    - Sauberes Spawning (max 2 Hindernisse)
    - Abstände grösser, aber nicht unfair
    """

    def __init__(self, seed: int | None = None):
        if seed is not None:
            random.seed(seed)

        # Fenster / Welt
        self.WIDTH = 900
        self.HEIGHT = 300
        self.GROUND_Y = 235
        self.DINO_X = 120

        # Physik
        self.GRAVITY = 1.15
        self.JUMP_VEL = -18.0
        self.SPEED = 9.0

        # Hindernisse (✅ grössere Abstände)
        self.MIN_GAP = 280
        self.MAX_GAP = 460

        # Random Kaktus-Grössen (nicht zu extrem)
        self.MIN_W = 18
        self.MAX_W = 55
        self.MIN_H = 20
        self.MAX_H = 55

        # Easy-Mode (Curriculum)
        self.training_easy = False
        self.EASY_W = 30
        self.EASY_H = 40

        # Render
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Dino Runner RL")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)

        self.render_enabled = True

        self.reset()

    def reset(self):
        self.done = False
        self.steps = 0
        self.cleared = 0

        # Dino
        self.dino_y = self.GROUND_Y
        self.dino_vy = 0.0
        self.on_ground = True

        # Hindernisse: immer 2
        self.obstacles = []
        self._spawn_obstacle(self.WIDTH + 300)
        self._spawn_obstacle()  # zweites mit Abstand

        return self.get_state()

    def _handle_quit(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

    def _spawn_obstacle(self, x: int | None = None):
        if self.training_easy:
            w = self.EASY_W
            h = self.EASY_H
        else:
            w = random.randint(self.MIN_W, self.MAX_W)
            h = random.randint(self.MIN_H, self.MAX_H)

        if x is None:
            last_x = self.obstacles[-1]["x"] if self.obstacles else self.WIDTH
            gap = random.randint(self.MIN_GAP, self.MAX_GAP)
            x = last_x + gap

        self.obstacles.append({"x": float(x), "w": int(w), "h": int(h), "passed": False})

    def _dino_rect(self) -> pygame.Rect:
        dino_w, dino_h = 42, 44
        return pygame.Rect(self.DINO_X, int(self.dino_y - dino_h), dino_w, dino_h)

    def _obs_rect(self, obs) -> pygame.Rect:
        return pygame.Rect(int(obs["x"]), int(self.GROUND_Y - obs["h"]), int(obs["w"]), int(obs["h"]))

    def _next_obstacle(self):
        candidates = [o for o in self.obstacles if o["x"] + o["w"] >= self.DINO_X]
        return min(candidates, key=lambda o: o["x"]) if candidates else self.obstacles[0]

    # ---------- State Binning (endlich, aber fein genug) ----------
    def _bin_dist(self, dist_px: float) -> int:
        """
        Fein im Nahbereich (0..300 in 10px steps -> 0..30),
        alles weiter weg = 31
        """
        if dist_px > 300:
            return 31
        b = int(dist_px // 10)
        return max(0, min(b, 30))

    def _bin_height(self, h: int) -> int:
        if h < 30: return 0
        if h < 45: return 1
        return 2

    def _bin_width(self, w: int) -> int:
        if w < 28: return 0
        if w < 42: return 1
        return 2

    def _bin_y(self) -> int:
        """
        Höhe des Dinos über Boden in Stufen -> hilft Timing.
        """
        above = self.GROUND_Y - self.dino_y  # 0 am Boden, positiv in der Luft
        if above < 5:
            return 0
        if above < 25:
            return 1
        if above < 55:
            return 2
        return 3

    def _bin_vy(self) -> int:
        """
        0 = am Boden/ruhig, 1 = aufwärts, 2 = abwärts
        """
        if self.on_ground:
            return 0
        return 1 if self.dino_vy < 0 else 2

    def get_state(self):
        nxt = self._next_obstacle()
        dist = max(0.0, nxt["x"] - self.DINO_X)

        dist_bin = self._bin_dist(dist)
        h_bin = self._bin_height(nxt["h"])
        w_bin = self._bin_width(nxt["w"])
        on_ground = 1 if self.on_ground else 0
        y_bin = self._bin_y()
        vy_bin = self._bin_vy()

        return (dist_bin, h_bin, w_bin, on_ground, y_bin, vy_bin)

    def step(self, action: int):
        if self.done:
            return self.get_state(), 0.0, True, {}

        # Bei Turbo-Training reicht Event-Check nicht jedes Frame
        if (self.steps % 15) == 0:
            self._handle_quit()

        reward = 0.10  # überleben

        # Action: springen nur am Boden
        if action == 1 and self.on_ground:
            self.dino_vy = self.JUMP_VEL
            self.on_ground = False
            reward -= 0.05  # kleine Strafe gegen Spam

        # Physik
        self.dino_vy += self.GRAVITY
        self.dino_y += self.dino_vy

        if self.dino_y >= self.GROUND_Y:
            self.dino_y = self.GROUND_Y
            self.dino_vy = 0.0
            self.on_ground = True

        # Hindernisse bewegen
        for obs in self.obstacles:
            obs["x"] -= self.SPEED

        # Entfernen wenn links raus
        self.obstacles = [o for o in self.obstacles if o["x"] + o["w"] > -80]

        # Immer 2 Hindernisse halten
        while len(self.obstacles) < 2:
            self._spawn_obstacle()

        self.steps += 1

        # Collision + passed
        dino_rect = self._dino_rect()
        for obs in self.obstacles:
            obs_rect = self._obs_rect(obs)

            if dino_rect.colliderect(obs_rect):
                self.done = True
                reward = -100.0
                break

            if not obs["passed"] and (obs["x"] + obs["w"] < self.DINO_X):
                obs["passed"] = True
                self.cleared += 1
                reward += 10.0

        next_state = self.get_state()
        info = {"steps": self.steps, "cleared": self.cleared}
        return next_state, reward, self.done, info

    def render(self, extra_text: str | None = None):
        if not self.render_enabled:
            return

        self.screen.fill((245, 245, 245))
        pygame.draw.line(self.screen, (30, 30, 30), (0, self.GROUND_Y), (self.WIDTH, self.GROUND_Y), 2)

        pygame.draw.rect(self.screen, (30, 150, 60), self._dino_rect())

        for obs in self.obstacles:
            pygame.draw.rect(self.screen, (40, 40, 40), self._obs_rect(obs))

        t1 = self.font.render(f"Cleared: {self.cleared}  Steps: {self.steps}", True, (10, 10, 10))
        self.screen.blit(t1, (10, 10))

        if extra_text:
            t2 = self.font.render(extra_text, True, (10, 10, 10))
            self.screen.blit(t2, (10, 32))

        pygame.display.flip()
