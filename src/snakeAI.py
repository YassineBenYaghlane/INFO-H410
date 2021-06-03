from gameModule import *
from copy import deepcopy
import heapq
import time
import argparse
import numpy as np
import multiprocessing

rng = np.random.default_rng(171)

SNAKE_CHAR = '+'
WALL_CHAR = '#'
FOOD_CHAR = '@'

### PARSER DEFINITION ###
#########################

parser = argparse.ArgumentParser(
    description="A* For Snake. If no argument is given, the game starts in player mode.")

group_play = parser.add_mutually_exclusive_group(required=False)
group_play.add_argument('-p', "--player", action='store_true', help="Player mode: the player controls the game")
group_play.add_argument('-x', "--ai", action='store_true', help="AI mode: the AI controls the game (requires an "
                                                                "algorithm argument)")
group_play.add_argument('-t', "--training", action='store_true',
                        help="Training mode: the AI controls the game and a "
                             "file is written to keep track of the scores ("
                             "requires an algorithm argument and an output "
                             "file)")
group_algorithm = parser.add_mutually_exclusive_group(required=False)
group_algorithm.add_argument('-r', "--random", action='store_true', help="Random play: a random move is drawn at "
                                                                         "iteration.")
group_algorithm.add_argument('-s', "--sshaped", action='store_true', help="S-Shaped algorithm: browses the whole "
                                                                          "grid each time in an 'S' shape. Only "
                                                                          "works if height of grid is even.")
group_algorithm.add_argument('-a', "--astar", action='store_true',
                             help="A* algorithm: classical A* algorithm, with "
                                  "Manhattan distance as heuristic")
group_algorithm.add_argument('-w', "--weighted", action='store_true',
                             help="Weighted A* algorithm: A* algorithm where "
                                  "the H value is weighted by a factor of 10")
group_algorithm.add_argument('-n', "--inverse", action='store_true',
                             help="Inverse A* algorithm: A* algorithm where "
                                  "the F value is 1000-F")
group_interactive = parser.add_mutually_exclusive_group(required=False)
group_interactive.add_argument('-i', "--interactive", action='store_true', help="Shows in a colorful way what the "
                                                                                "algorithm is computing (only useful "
                                                                                "for A* "
                                                                                "and variants)")

parser.add_argument('-z', "--survival", action='store_true',
                    help="use survival mode if specified")

parser.add_argument('-o', "--output", type=str,
                    help="To specify the file to write the results of the training in. If "
                         "specified in another mode, no file will be created")

args = parser.parse_args()

if (args.ai or args.training) and not (not args.sshaped or not args.astar or
                                       not args.weighted or not args.inverse):
    parser.error("AI or Training mode must be precised an algorithm.")

if args.training and not args.output:
    parser.error("An output filename must be specified in training mode.")

if args.interactive:
    interactive = True
else:
    interactive = False


def main():

    class Node:
        def __init__(self, position, parent):
            self.position = position
            self.parent = parent

            self.g = 0
            self.h = 0
            self.f = self.g + self.h

        def __repr__(self):
            return f'({self.position[0]}, {self.position[1]})'

        def __lt__(self, other):
            return self.f < other.f

        def __eq__(self, other):
            if isinstance(other, Node):
                return self.position == other.position
            else:
                return self.position == other

    class IAExample:
        def __init__(self):
            self.moves = [RIGHT, DOWN, LEFT, UP]
            self.best_path = None # The path used by the snake
            self.first = True # Boolean
            self.is_in_survival_mode = False

        def choose_next_move(self, state):
            """
                This function is called by the game instance in order to find the next move chosen by the used algorithm.
                In our case, there are 5 different algorithms : Random, SShaped, A*, A* weighted and A* reversed. All of
                them are described in detail in the report

            :param state: The state containing the grid, the snake body, the score and a boolean indicating if the snake
                          is alive
            :return: The move chosen by the algorithm
            """

            grid, score, alive, snake = state
            head = snake[0]

            # args.survival is the boolean that indicates if the user wanted the algorithm to use the survival mode
            # self.is_in_survival_mode indicates if the algorithm is in survival mode at the current move
            if args.survival:
                # check if the snake can find a path to the apple and then end survival mode
                if self.is_in_survival_mode and self.best_path and self.astar(state, game.food, mode='default', interactive=interactive) != 171:
                    self.best_path = []
                    print("End Survival mode")
                    self.is_in_survival_mode = False

            # Random algorithm
            if args.random:
                r = rng.integers(4)
                next_move = self.moves[r]

            else:
                # When self.best_path is empty, that means we need to generate a path with the algorithm specified.
                # We also need to end the survival mode if the snake was previously in this mode
                if not self.best_path:
                    if args.survival:
                        if self.is_in_survival_mode:
                            print("End Full Survival mode")
                            self.is_in_survival_mode = False
                    if args.astar:
                        self.best_path = self.astar(state, game.food, mode='default', interactive=interactive)
                    elif args.weighted:
                        self.best_path = self.astar(state, game.food, mode='weighted', interactive=interactive)
                    elif args.inverse:
                        self.best_path = self.astar(state, game.food, mode='inverse', interactive=interactive)
                    elif args.sshaped:
                        self.best_path = self.sshape(state)

                # When A* does not find any path to his goal, our implementation returns 171. In that case, we need to
                # enter in survival mode (if specified by the user).
                if self.best_path == 171:
                    if args.survival:
                        print('Start Survival mode')
                        self.is_in_survival_mode = True
                        # time.sleep(5)
                        self.best_path = self.survival_mode(state)
                        if self.best_path == 171:
                            print("Survival mode did not work")
                            return self.moves[1]
                    else:
                        # if the user does not specify to use the survival mode, the snake goes DOWN when A* does not work
                        print("A* did not find path")
                        return self.moves[1]

                next_move = self.get_next_move(self.best_path, head)

            return next_move



        def get_next_move(self, path, head):
            """
                This function finds the next move to do based on the head and the path.
            :param path: The path followed by the snake
            :param head: The head of the snake
            :return: The next move
            """
            next_node = path.pop()
            next_pos = next_node.position
            next_move = self.moves[0]

            next_mov_bool = []

            for i in range(len(next_pos)):
                next_mov_bool.append(next_pos[i] - head[i])

            if next_mov_bool[0] == 0:
                if next_mov_bool[1] == 1:
                    next_move = self.moves[0]
                elif next_mov_bool[1] == -1:
                    next_move = self.moves[2]
                else:
                    print("Problem in moves, head: ", head, ", next_pos: ", next_pos)
            elif next_mov_bool[0] == 1:
                next_move = self.moves[1]
            elif next_mov_bool[0] == -1:
                next_move = self.moves[3]
            else:
                print("Problem in moves, head: ", head, ", next_pos: ", next_pos)

            return next_move

        def h_cost(self, current, end):
            """
                Cost used in the A* algorithm
            :param current: current node
            :param end: end node
            :return: the Manhattan distance between the current node and the end node
            """
            res = abs(current.position[0] - end.position[0]) + abs(
                current.position[1] - end.position[1])
            return res

        def dist_to_snake(self, current, snake):
            """
                This function computes the minimum distance between a node and the snake body
            :param current: current node
            :param snake: snake body
            :return:
            """
            best_cost = 100000
            for i in snake:
                n = Node(i, None)
                cost = self.h_cost(current, n)
                if cost < best_cost:
                    best_cost = cost
            return best_cost

        def astar(self, state, goal_pos, mode='default', interactive=False):
            """
                This function is an implementation of the A* algorithm
            :param state: The current state of the game
            :param goal_pos: The position where the snake has to go
            :param mode: default = classic A*, weighted = weighted A*, inverse = reverse A*, survival = A* for survival mode
            :param interactive: Display the execution of the A* algorithm
            :return: The path to the goal
            """
            grid, score, alive, snake = state
            head = snake[0]
            closed_list = []
            open_list = []
            head_node = Node(head, None)
            food_node = Node(goal_pos, None)

            if mode == "survival":
                game.grid[food_node.position[0]][food_node.position[1]] = EMPTY_CHAR

            heapq.heappush(open_list, head_node)

            while open_list:
                current_node = heapq.heappop(open_list)
                closed_list.append(current_node)

                if interactive:
                    time.sleep(0.1)
                    game.grid[current_node.position[0]][current_node.position[1]] = 'C'
                    game.draw()

                if current_node.position == food_node.position:
                    path = []
                    while current_node.parent is not None:
                        path.append(current_node)
                        current_node = current_node.parent

                    if interactive:
                        for el in path:
                            game.grid[el.position[0]][el.position[1]] = 'A'
                            time.sleep(0.1)
                            game.draw()
                        for el in path:
                            game.grid[el.position[0]][el.position[1]] = ' '
                        for el in open_list:
                            game.grid[el.position[0]][el.position[1]] = ' '
                        for el in closed_list:
                            game.grid[el.position[0]][el.position[1]] = ' '
                        game.grid[food_node.position[0]][food_node.position[1]] = FOOD_CHAR
                        game.grid[head_node.position[0]][head_node.position[1]] = SNAKE_CHAR
                        game.draw()

                    if mode == "survival":
                        game.grid[food_node.position[0]][food_node.position[1]] = SNAKE_CHAR
                    return path

                children = []
                for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                    node_position = (
                        current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

                    # Make sure within range
                    if node_position[0] > (len(grid) - 1) or node_position[0] < 0 or node_position[1] > (
                            len(grid[len(grid) - 1]) - 1) or node_position[1] < 0:
                        continue

                    # Make sure walkable terrain
                    if grid[node_position[0]][node_position[1]] == SNAKE_CHAR or \
                            grid[node_position[0]][node_position[1]] == WALL_CHAR:
                        continue

                    # Create new node
                    new_node = Node(node_position, current_node)

                    # Append
                    children.append(new_node)

                # Loop through children
                for child in children:

                    # Child is on the closed list
                    if child in closed_list:
                        continue

                    if child in open_list:
                        if child.g > current_node.g + 1:
                            child.g = current_node.g + 1
                            child.parent = current_node

                    else:
                        # Create the f, g, and h values
                        if mode == 'default':
                            child.g = current_node.g + 1
                            child.h = self.h_cost(child, food_node)
                            child.f = child.g + child.h

                        elif mode == 'weighted':
                            child.g = current_node.g + 1
                            child.h = 10 * (self.h_cost(child, food_node))
                            child.f = child.g + child.h

                        elif mode == 'inverse':
                            child.g = current_node.g + 1
                            child.h = self.h_cost(child, food_node)
                            child.f =  10000 - (child.g + child.h)

                        elif mode == 'survival':
                            child.g = current_node.g + 1
                            child.h = self.h_cost(child, food_node)
                            child.f = - (child.g + 5 * child.h) + 2 * self.dist_to_snake(child, snake)

                        child.parent = current_node

                        # Add the child to the open list
                        heapq.heappush(open_list, child)

                        if interactive:
                            if game.grid[child.position[0]][child.position[1]] == '+':
                                pass
                            else:
                                game.grid[child.position[0]][child.position[1]] = 'S'
                            game.draw()

            if mode == "survival":
                game.grid[food_node.position[0]][food_node.position[1]] = SNAKE_CHAR

            return 171

        def sshape(self, state):
            """
                SShaped implementation. The snake does the same path during all the game and it gives the perfect score.
            :param state: The current state of the game
            :return: The path the snake has to follow
            """

            grid, score, alive, snake = state
            head = snake[0]
            path = []
            if score == 0:
                if head[0] == 0 and self.first:
                    path.append(Node((head[0] + 1, head[1]), None))
                    self.first = False
                    return path

                else:
                    self.first = False

                for i in range(head[1] + 1, len(grid[1])):
                    path.append(Node((head[0], i), None))

                for i in range(1, head[0] + 1):
                    path.append(Node((head[0] - i, head[1]), None))

            for i in range(len(grid)):
                if i % 2 == 1:
                    for j in range(len(grid[0]) - 1):
                        path.append(Node((i, j), None))
                else:
                    for j in range(len(grid[0]) - 2, -1, -1):
                        path.append(Node((i, j), None))

            for i in range(len(grid)):
                path.append(Node((len(grid) - i - 1, len(grid[0]) - 1), None))

            return path[::-1]

        def survival_mode(self, state):
            """
                This function is a Survival mode implementation. The snake tries to find the longest path from his head
                to his tail beginning at the last node of his tail to the 5th node.
            :param state: The current state of the game
            :return: The longest path to his tail. The function may return 171 which indicate that no path has been found.
                     This case is managed at a higher level.
            """
            grid, score, alive, snake = state
            last_attainable_node_index = len(snake) - 1
            best_path = 171

            while last_attainable_node_index > 5:
                path = self.astar(state, snake[last_attainable_node_index], mode="survival")
                if path != 171 and len(path) >= 3 and ((best_path != 171 and len(path) > len(best_path)) or (best_path == 171)):
                    best_path = path
                last_attainable_node_index -= 1


            return best_path


    agent = IAExample() if (args.ai or args.training) else None  # None for interactive GUI

    ### FOR TRAINING : ###
    ######################

    if args.training:
        out_file = open(args.output.split()[-1], 'w')
        out_file.write('count,score\n')
        game = TrainingSnakeGame(agent)
        game.start_run()
        start_time = time.time()

        count = 0
        while game.is_alive() and game.score < 100:
            out_file.write(str(count)+','+str(game.score)+'\n')
            game.next_tick()
            count += 1
        print('Game not alive')
        time_taken = time.time() - start_time
        out_file.flush()
        out_file.write(str(count)+','+format(time_taken, '.4f'))
        out_file.flush()
        out_file.close()

    ### FOR GAMING : ###
    #####################

    else:
        game = GUISnakeGame()
        game.init_pygame()

        while game.is_running():
            game.next_tick(agent)

        game.cleanup_pygame()

main()
