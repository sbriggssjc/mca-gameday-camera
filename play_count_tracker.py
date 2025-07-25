# A simple play count tracker for 17 players.
# Each play, enter the players who participated (numbers 1-17) separated by spaces.
# Enter 'q' to quit. The script logs plays to play_log.csv and prints counts.

import csv


def main():
    play_counts = {i: 0 for i in range(1, 18)}
    play_number = 1

    with open('play_log.csv', 'w', newline='') as logfile:
        writer = csv.writer(logfile)
        writer.writerow(['play_number', 'players'])

        while True:
            user_input = input(f"Enter player numbers for play {play_number} (or 'q' to quit): ")
            if user_input.lower() in {'q', 'quit'}:
                break
            try:
                players = [int(x) for x in user_input.split()]
            except ValueError:
                print('Invalid input. Use numbers 1-17 separated by spaces.')
                continue
            invalid = [p for p in players if p < 1 or p > 17]
            if invalid:
                print(f'Invalid player numbers: {invalid}. Valid range is 1-17.')
                continue
            writer.writerow([play_number, ' '.join(str(p) for p in players)])
            for p in set(players):
                play_counts[p] += 1
            play_number += 1

    print('\nPlay Counts:')
    for player in range(1, 18):
        count = play_counts[player]
        if count < 7:
            print(f'Player {player:2d}: {count} < 7! ALERT')
        else:
            print(f'Player {player:2d}: {count}')


if __name__ == '__main__':
    main()
