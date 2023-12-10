#By Reese Johnson 

# importing random package
# for methods to generate random
# numbers.
import random

# function to initialize game / grid
# at the start
def start_game():

	# declaring an empty list then
	# appending 4 list each with four
	# elements as 0.
	mat =[]
	for i in range(4):
		mat.append([0] * 4)

	# printing controls for user
	# print("Commands are as follows : ")
	# print("'W' or 'w' : Move Up")
	# print("'S' or 's' : Move Down")
	# print("'A' or 'a' : Move Left")
	# print("'D' or 'd' : Move Right")

	# calling the function to add
	#initialize board with 2 numbers on grid
	add_random_tile(mat)
	add_random_tile(mat)
	return mat

# function to add a new 2 in
# grid at any random empty cell
#old: def add_new_2(mat):
def add_random_tile(mat):

	# choosing a random index for
	# row and column.
	r = random.randint(0, 3)
	c = random.randint(0, 3)

	# while loop will break as the
	# random cell chosen will be empty
	# (or contains zero)
	while(mat[r][c] != 0):
		r = random.randint(0, 3)
		c = random.randint(0, 3)

	# we will place a 2 at that empty
	# random cell.
	if random.random() < 0.9:
		#generates a 2 90% of the time
		mat[r][c] = 2
	
	else:
		mat[r][c] = 4

# function to get the current
# state of game

def is_game_over(mat):

	moves = get_all_moves(mat)

	if len(moves) == 0:
		return True
	else:
		return False
	
def get_all_moves(mat):
	
	moves = []
	
	#checking left move attempt
	#i = rows
	found_move = False
	for i in range(4):
		#empty row cant move left
		#exclude farthest left cloumn b/c cant move left anyway
		for j in range(1,4):
			#if theres a tile in the row, and theres empty space or same number to left then ok
			if mat[i][j] != 0 and (mat[i][j-1] == mat[i][j] or mat[i][j-1] == 0):
				moves.append("left")
				found_move = True
				break
		if found_move:
			break
	
	#checking right move attempt
	#i = rows
	found_move = False
	for i in range(4):
		#empty row cant move left
		#exclude farthest left cloumn b/c cant move left anyway
		for j in range(0,3):
			#if theres a tile in the row, and theres empty space or same number to right then ok
			if mat[i][j] != 0 and (mat[i][j+1] == mat[i][j] or mat[i][j+1] == 0):
				moves.append("right")
				found_move = True
				break
		if found_move:
			break
	
	#checking up move attempt
	#i = rows, j = columns, iterate xc rows, down columns
	found_move = False
	for j in range(4):
		#empty row cant move left
		#exclude farthest left cloumn b/c cant move left anyway
		for i in range(1,4):
			#if theres a tile in the row, and theres empty space or same number to up then ok
			if mat[i][j] != 0 and (mat[i-1][j] == mat[i][j] or mat[i-1][j] == 0):
				moves.append("up")
				found_move = True
				break
		if found_move:
			break

	#checking down move attempt
	#i = rows, j = columns, iterate xc rows, down columns
	found_move = False
	for j in range(4):
		#empty row cant move left
		#exclude farthest left cloumn b/c cant move left anyway
		for i in range(0,3):
			#if theres a tile in the row, and theres empty space or same number to up then ok
			if mat[i][j] != 0 and (mat[i+1][j] == mat[i][j] or mat[i+1][j] == 0):
				moves.append("down")
				found_move = True
				break
		if found_move:
			break

	return moves


def move_left(mat):
	#first condense spaces
	for i in range(4):
		for j in range(1,4):
			if mat[i][j-1] == 0:
				mat[i][j-1] = mat[i][j]
				mat[i][j] = 0

		for j in range(1,3):
			if mat[i][j-1] == 0:
				mat[i][j-1] = mat[i][j]
				mat[i][j] = 0

		for j in range(1,2):
			if mat[i][j-1] == 0:
				mat[i][j-1] = mat[i][j]
				mat[i][j] = 0

		#now see if addition needs to be done
		for j in range(1,4):
			if mat[i][j-1] == mat[i][j]:
				mat[i][j-1] *= 2
				mat[i][j] = 0
				#need to eliminate one that didnt get doubled, and then move down numbers, fill w 0s
				for k in range(j,3):
					mat[i][k] = mat[i][k+1]
					mat[i][k+1] = 0

	
def move_right(mat):
	#first condense spaces
	for i in range(4):
		for j in range(2, -1, -1):
			if mat[i][j+1] == 0:
				mat[i][j+1] = mat[i][j]
				mat[i][j] = 0

		for j in range(2,0,-1):
			if mat[i][j+1] == 0:
				mat[i][j+1] = mat[i][j]
				mat[i][j] = 0

		for j in range(2,1,-1):
			if mat[i][j+1] == 0:
				mat[i][j+1] = mat[i][j]
				mat[i][j] = 0

		#now see if addition needs to be done
		for j in range(2,-1,-1):
			if mat[i][j+1] == mat[i][j]:
				mat[i][j+1] *= 2
				mat[i][j] = 0
				#need to eliminate one that didnt get doubled, and then move down numbers, fill w 0s
				for k in range(j,0):
					mat[i][k] = mat[i][k-1]
					mat[i][k-1] = 0

#up is going to be like left

def move_up(mat):
	#first condense spaces
	for j in range(4):
		for i in range(1,4):
			if mat[i-1][j] == 0:
				mat[i-1][j] = mat[i][j]
				mat[i][j] = 0

		for i in range(1,3):
			if mat[i-1][j] == 0:
				mat[i-1][j] = mat[i][j]
				mat[i][j] = 0

		for i in range(1,2):
			if mat[i-1][j] == 0:
				mat[i-1][j] = mat[i][j]
				mat[i][j] = 0

		#now see if addition needs to be done
		for i in range(1,4):
			if mat[i-1][j] == mat[i][j]:
				mat[i-1][j] *= 2
				mat[i][j] = 0
				#need to eliminate one that didnt get doubled, and then move down numbers, fill w 0s
				for k in range(i,3):
					mat[k][j] = mat[k+1][j]
					mat[k+1][j] = 0


def move_down(mat):
	#first condense spaces
	for j in range(4):
		for i in range(2, -1, -1):
			if mat[i+1][j] == 0:
				mat[i+1][j] = mat[i][j]
				mat[i][j] = 0

		for i in range(2,0,-1):
			if mat[i+1][j] == 0:
				mat[i+1][j] = mat[i][j]
				mat[i][j] = 0

		for i in range(2,1,-1):
			if mat[i+1][j] == 0:
				mat[i+1][j] = mat[i][j]
				mat[i][j] = 0

		#now see if addition needs to be done
		for i in range(2,-1,-1):
			if mat[i+1][j] == mat[i][j]:
				mat[i+1][j] *= 2
				mat[i][j] = 0
				#need to eliminate one that didnt get doubled, and then move down numbers, fill w 0s
				for k in range(i,0):
					mat[k][j] = mat[k-1][j]
					mat[k-1][j] = 0

def simulate_game(policy, show_board=False, show_score=False):
	matrix = start_game()
	moves = get_all_moves(matrix)

	if show_score:
		score = game_score(matrix)
		print("SCORE: ", score)

	
	if show_board:
		for row in matrix:
			print(row)
		print(" ")

	while len(moves) != 0:

		action = policy(matrix, moves)

		if action == "up":
			move_up(matrix)
		
		if action == "down":
			move_down(matrix)
		
		if action == "left":
			move_left(matrix)
		
		if action == "right":
			move_right(matrix)
	
		add_random_tile(matrix)
		moves = get_all_moves(matrix)

		if show_score:
			score = game_score(matrix)
			print("SCORE: ", score)
		
		if show_board:
			for row in matrix:
				print(row)
			print(" ")

	print("YOU LOSE")


score_lookup = {0:0, 2:0, 4:4, 
				8:16, 16:48, 32:128, 
				64:320, 128:768, 256:1792, 
				512:4096, 1024:9216, 2048:20480, 
				4096:45056, 8192:98304, 16384:212992, 
				32768:458752, 64536: 983040, 131072: 2097152}

def game_score(matrix):
	
	score = 0
	for i in range(4):
		for j in range(4):
			score += score_lookup[matrix[i][j]]
	
	return score
	
def get_successor_state(matrix, action):
	successor = []

	#difficult because of non deterministic successor states, go with random
	for i in range(4):
		successor.append([0,0,0,0])
		for j in range(4):
			successor[i][j] = matrix[i][j]
	
	if action == "up":
		move_up(successor)
	
	if action == "down":
		move_down(successor)
	
	if action == "left":
		move_left(successor)
	
	if action == "right":
		move_right(successor)

	add_random_tile(successor)

	return successor


###POLICIES
def human_interaction_policy(matrix, moves):
	
	for row in matrix:
		print(row)
	
	print(moves)
	
	selected_action = ""
	while selected_action not in moves:
		selected_action = input("Select a move: ")

	return selected_action
# simulate_game(human_interaction_policy, show_board=False, show_score=True)


def random_player(matrix, moves):
	return random.choice(moves)
# simulate_game(random_player, show_board=True, show_score=True)


#greedy player always picks the move that will increase his score the most at any given game state
def greedy_player(matrix, moves):

	max_score = -1
	optimal_action = ""

	for move in moves:
		successor = get_successor_state(matrix, move)
		score = game_score(successor)
		
		if score > max_score:
			max_score = score
			optimal_action = move

	return optimal_action

simulate_game(greedy_player, show_board=True, show_score=True)

