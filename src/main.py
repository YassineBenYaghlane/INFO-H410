from game import *
import heapq
import time
import argparse

SNAKE_CHAR = '+'
EMPTY_CHAR = ' '
WALL_CHAR = '#'
FOOD_CHAR = '@'

parser = argparse.ArgumentParser(description="A* For Snake")
group_solution = parser.add_mutually_exclusive_group(required=False)
group_solution.add_argument('-i', "--interactive", action='store_true', help="Returns solution as a vector instead of the sum")
args = parser.parse_args()

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
            return f'Node f value: {self.f}'

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

        def choose_next_move(self, state):
            grid, score, alive, head = state
            # print("Choosing next move")

            if not self.best_path:
                self.best_path = self.astar(state, interactive)

            if self.best_path == 171:
                print("A Star marche pas")
                return self.moves[1]

            next_node = self.best_path.pop()
            next_pos = next_node.position



            next_mov_bool = []

            for i in range(len(next_pos)):
                next_mov_bool.append(next_pos[i]-head[i])
            #print(next_mov_bool)

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
                print("Problem in moves, head: ",head,", next_pos: ",next_pos)

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


        def astar(self, state, interactive=False):
            print("Starting A* search")
            grid, score, alive, head = state
            closed_list = []
            open_list = []
            head_node = Node(head, None)
            food_node = Node(game.food, None)
            heapq.heappush(open_list, head_node)

            while open_list:
                current_node = heapq.heappop(open_list)
                # print("Current Node: ", current_node)
                closed_list.append(current_node)

                if interactive:
                    game.grid[current_node.position[0]][current_node.position[1]] = 'C'
                    game.draw()

                if current_node.position == food_node.position:
                    path = []
                    while current_node.parent is not None:
                        path.append(current_node)
                        current_node = current_node.parent

                    if interactive:
                        for el in path:
                            print(el.position)
                            game.grid[el.position[0]][el.position[1]] = 'A'
                            time.sleep(0.1)
                            game.draw()
                        for el in path:
                            print(el.position)
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
                    node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

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

            print("ON SAIT PAS")
            return 171

    agent = IAExample()  # None for interactive GUI
    game = GUISnakeGame()
    game.init_pygame()

    while game.is_running():
        game.next_tick(agent)

    game.cleanup_pygame()

    #game = TrainingSnakeGame(agent)
    #game.start_run()

    #while game.is_alive():
    #    game.next_tick()


#if __name__ == '__main__':
main()
