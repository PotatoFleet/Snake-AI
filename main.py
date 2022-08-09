from random import randint
import pygame
import math
import neat

pygame.font.init()

WIDTH, HEIGHT = 600, 600

SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))

pygame.display.set_caption("Snake AI")

FONT = pygame.font.SysFont("segoeui", 32)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 100, 0)
RED = (255, 0, 0)

UPDATE = pygame.USEREVENT + 1

pygame.time.set_timer(UPDATE, 50)

generation = 0


class Grid:
    def __init__(self):
        self.squares = [[Square(i, j)
                         for i in range(WIDTH // Square.SQUARE_WIDTH)] for j in range(HEIGHT // Square.SQUARE_WIDTH)]

    def render(self, snake, food):
        for i in range(len(self.squares)):
            for j in range(len(self.squares[i])):
                if [j, i] in snake:
                    if snake[0] == [j, i]:
                        self.squares[i][j].render(DARK_GREEN)
                    else:
                        self.squares[i][j].render(GREEN)
                elif [j, i] == food:
                    self.squares[i][j].render(RED)
                else:
                    self.squares[i][j].render(WHITE)


class Square:
    SQUARE_WIDTH = 20

    def __init__(self, x, y):
        self.x = x * Square.SQUARE_WIDTH
        self.y = y * Square.SQUARE_WIDTH
        self.color = BLACK
        self.rect = pygame.Rect(
            self.x, self.y, self.x + Square.SQUARE_WIDTH, self.y + Square.SQUARE_WIDTH)

    def render(self, color=None):
        pygame.draw.rect(
            SCREEN, color, self.rect)


def random_square(snake):
    res = [randint(0, WIDTH // Square.SQUARE_WIDTH - 1),
           randint(0, HEIGHT // Square.SQUARE_WIDTH - 1)]
    while res in snake:
        res = [randint(0, WIDTH // Square.SQUARE_WIDTH - 1),
               randint(0, HEIGHT // Square.SQUARE_WIDTH - 1)]
    return res


def render(grid, snake, food, score):
    SCREEN.fill(WHITE)
    grid.render(snake, food)
    text = FONT.render("Score: " + str(score), True, BLACK)
    SCREEN.blit(text, (10, 10))
    text = FONT.render("Gen: " + str(generation), True, BLACK)
    SCREEN.blit(text, (10, 50))


def run(genomes, config):
    global generation
    nets = []
    ge = []

    for i, g in genomes:
        g.fitness = 0
        ge.append(g)
        nets.append(neat.nn.FeedForwardNetwork.create(g, config))

    grid = [Grid() for _ in range(len(ge))]
    snake = [[[15, 15]] for _ in range(len(ge))]
    snake_len = [3 for _ in range(len(ge))]
    food = [random_square(snake[i]) for i in range(len(ge))]
    score = [0 for _ in range(len(ge))]
    count = [0 for _ in range(len(ge))]
    '''
    Directions:
    1 - Up
    2 - Right
    3 - Down
    4 - Left
    '''
    direction = [randint(1, 4) for _ in range(len(ge))]
    high_score = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == UPDATE:
                heads = [s[0].copy() for s in snake]
                i = 0
                while i < len(ge):
                    if direction[i] == 1:
                        heads[i][1] -= 1
                    elif direction[i] == 2:
                        heads[i][0] += 1
                    elif direction[i] == 3:
                        heads[i][1] += 1
                    elif direction[i] == 4:
                        heads[i][0] -= 1

                    if heads[i][0] < 0 or \
                            heads[i][1] < 0 or \
                            heads[i][0] == WIDTH // Square.SQUARE_WIDTH or \
                            heads[i][1] == HEIGHT // Square.SQUARE_WIDTH or \
                            heads[i] in snake[i] or count[i] == 100:
                        ge[i].fitness -= 40
                        ge.pop(i)
                        heads.pop(i)
                        snake.pop(i)
                        direction.pop(i)
                        score.pop(i)
                        food.pop(i)
                        nets.pop(i)
                        continue

                    ge[i].fitness -= 1
                    count[i] += 1

                    d1 = math.sqrt(
                        (snake[i][0][0] - food[i][0]) ** 2 + (snake[i][0][1] - food[i][1]) ** 2)

                    snake[i].insert(0, heads[i])

                    d2 = math.sqrt(
                        (snake[i][0][0] - food[i][0]) ** 2 + (snake[i][0][1] - food[i][1]) ** 2)

                    if d2 > d1:
                        ge[i].fitness -= 2
                    else:
                        ge[i].fitness += 2

                    if snake[i][0] == food[i]:
                        food[i] = random_square(snake[i])
                        score[i] += 1
                        high_score = max(high_score, score[i])
                        ge[i].fitness += 50
                        snake_len[i] += 1
                        count[i] = 0
                    elif len(snake[i]) > snake_len[i]:
                        snake[i].pop(-1)

                    head_food_distance_x = snake[i][0][0] - food[i][0]
                    head_food_distance_y = snake[i][0][1] - food[i][1]
                    head_left_wall_distance = snake[i][0][0]
                    head_right_wall_distance = WIDTH // Square.SQUARE_WIDTH - \
                        snake[i][0][0]
                    head_top_wall_distance = snake[i][0][1]
                    head_bottom_wall_distance = HEIGHT // Square.SQUARE_WIDTH - \
                        snake[i][0][1]

                    top = [snake[i][0][0], snake[i][0][1] - 1] in snake[i][1:]
                    left = [snake[i][0][0] - 1, snake[i][0][1]] in snake[i][1:]
                    right = [snake[i][0][0] + 1,
                             snake[i][0][1]] in snake[i][1:]
                    bottom = [snake[i][0][0], snake[i]
                              [0][1] + 1] in snake[i][1:]

                    outputs = nets[i].activate((
                        1 if top else 0, 1 if left else 0, 1 if right else 0, 1 if bottom else 0,
                        1 if direction[i] == 1 else 0, 1 if direction[i] == 2 else 0, 1 if direction[
                            i] == 3 else 0, 1 if direction[i] == 4 else 0,
                        head_food_distance_x, head_food_distance_y, head_left_wall_distance,
                        head_right_wall_distance, head_top_wall_distance, head_bottom_wall_distance))

                    output = outputs.index(max(outputs)) + 1
                    # if abs(output - direction[i]) == 2:
                    #     ge[i].fitness -= 5
                    direction[i] = output
                    i += 1

        if len(ge) == 0:
            break

        render(grid[0], snake[0], food[0], score[0])
        pygame.display.update()


    generation += 1
    print("GEN " + str(generation) + " - HIGH SCORE: " + str(high_score))


def main():
    config_path = "./config"
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    p = neat.Population(config)

    # p.add_reporter(neat.StdOutReporter(True))
    # p.add_reporter(neat.StatisticsReporter())

    p.run(run, 500)


if __name__ == "__main__":
    main()
