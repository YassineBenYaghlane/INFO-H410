from gameModule import *
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
            self.i = 0
            self.best_path = None
            self.first = True

        def choose_next_move(self, state):
            grid, score, alive, head = state
            # print("Choosing next move")

            if args.random:
                r = rng.integers(4)
                next_move = self.moves[r]

            else:
                if not self.best_path:
                    if args.astar:
                        self.best_path = self.astar(state, mode='default', interactive=interactive)
                    elif args.weighted:
                        self.best_path = self.astar(state, mode='weighted', interactive=interactive)
                    elif args.inverse:
                        self.best_path = self.astar(state, mode='inverse', interactive=interactive)
                    elif args.sshaped:
                        self.best_path = self.sshape(state)

                if self.best_path == 171:
                    print('A* did not find any path')
                    game.alive = False
                    return self.moves[1]

                next_node = self.best_path.pop()
                next_pos = next_node.position

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

        def h_cost(self, current, end, grid):
            res = abs(current.position[0] - end.position[0]) + abs(
                current.position[1] - end.position[1])

            # print(f"grid ({len(grid)},{len(grid[0])})")
            # print(f"current : ({current.position[0]},{current.position[1]})")
            # dist_to_border_x = min(current.position[1], len(grid[0]) - current.position[1])
            # dist_to_border_y = min(current.position[0], len(grid) - current.position[0])
            # print(f"dist to border x : {dist_to_border_x}")
            # print(f"dist to border y : {dist_to_border_y}")
            # res += dist_to_border_x*10 + dist_to_border_y*10
            return res

        def astar(self, state, mode='default', interactive=False):
            grid, score, alive, head = state
            closed_list = []
            open_list = []
            head_node = Node(head, None)
            food_node = Node(game.food, None)
            heapq.heappush(open_list, head_node)

            while open_list:
                current_node = heapq.heappop(open_list)
                closed_list.append(current_node)

                if interactive:
                    time.sleep(0.2)
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
                            time.sleep(0.2)
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
                            child.h = self.h_cost(child, food_node, grid)
                            child.f = child.g + child.h

                        elif mode == 'weighted':
                            child.g = current_node.g + 1
                            child.h = 10 * (self.h_cost(child, food_node, grid))
                            child.f = child.g + child.h

                        elif mode == 'inverse':
                            child.g = current_node.g + 1
                            child.h = self.h_cost(child, food_node, grid)
                            child.f = 100000 - (child.g + child.h)

                        child.parent = current_node

                        # Add the child to the open list
                        heapq.heappush(open_list, child)

                        if interactive:
                            if game.grid[child.position[0]][child.position[1]] == '+':
                                pass
                            else:
                                game.grid[child.position[0]][child.position[1]] = 'S'
                            game.draw()

            return 171

        def sshape(self, state):
            grid, score, alive, head = state
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


#if __name__ == '__main__':
main()
