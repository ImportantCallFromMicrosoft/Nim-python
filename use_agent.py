import pygame

from agent_definition import NimAction, NimGameEnvironment, NimGameState, NimAgent


WIDTH = 500
HEIGHT = 500

def get_valid_action_from_agent(
    env: NimGameEnvironment, agent: NimAgent
):
    invalid = True
    while invalid:
        action = NimAction.from_idx(agent.get_action(hash(env.state)))
        invalid = not env.action_valid(action)
    return action

def draw_game_state(screen, game_state: NimGameState):
    screen.fill((0, 0, 0))
    for col in range(5):
        for row in range(col + 1):
            # fill the box
            if game_state[col] > row:
                pygame.draw.rect(
                    screen,
                    (255, 255, 255),
                    (col * 100, HEIGHT - 100 - row * 100, 100, 100),
                )
                pygame.draw.rect(
                    screen,
                    (0, 0, 255),
                    (col * 100, HEIGHT - 100 - row * 100, 100, 100),
                    3,
                )
    pygame.display.flip()


def show_victory_screen(screen):
    font = pygame.font.SysFont("Arial", 50)
    text = font.render("You win!", True, (0, 255, 0))

    # draw the text in the center of the screen
    text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    screen.blit(text, text_rect)
    pygame.display.flip()


def show_defeat_screen(screen):
    font = pygame.font.SysFont("Arial", 50)
    text = font.render("You lose!", True, (255, 0, 0))

    # draw the text in the center of the screen
    text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    screen.blit(text, text_rect)
    pygame.display.flip()


def show_invalid_move(screen):
    font = pygame.font.SysFont("Arial", 50)
    text = font.render("Illegal move!", True, (255, 0, 0))

    # draw the text in the center of the screen
    text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    screen.blit(text, text_rect)
    pygame.display.flip()
    # wait for 1 second
    pygame.time.wait(1000)


def main():
    agent = NimAgent.load("agent.json")
    env = NimGameEnvironment()
    env.reset()

    pygame.init()
    screen = pygame.display.set_mode((500, 500))
    pygame.display.set_caption("Draw Boxes")
    clock = pygame.time.Clock()

    draw_game_state(screen, env.state)
    pygame.display.flip()

    running = True
    while running:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                # User plays first
                x, y = pygame.mouse.get_pos()
                slot = x // 100
                pos_y = 5 - (y // 100) 
                amount = env.state[slot] - pos_y
                
                player_action = NimAction(slot, amount)

                if not env.action_valid(player_action):
                    show_invalid_move(screen)
                    draw_game_state(screen, env.state)
                    break

                print(player_action)
                _, _, lost, _, _ = env.step(player_action)
                print(env.state)
                draw_game_state(screen, env.state)

                if lost:
                    show_defeat_screen(screen)
                    running = False
                    break

                # Agent plays second
                
                agent_action = get_valid_action_from_agent(env, agent)
                print(agent_action)
                _, _, won, _, _ = env.step(agent_action)
                print(env.state)
                draw_game_state(screen, env.state)

                if won:
                    show_victory_screen(screen)
                    running = False
                    break

    # Wait for user to close window
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return


if __name__ == "__main__":
    main()
