import json
import sys

def arg_max(dictionary):
    v = list(dictionary.values())
    k = list(dictionary.keys())
    return k[v.index(max(v))]

def get_emission_prob(emission_prob, word, state):

    if word not in emission_prob.keys():
        return 0
    if state not in emission_prob[word].keys():
        return 1
    return 2

def hmmcode(line, transition_prob, emission_prob, open_class):

    states = list(transition_prob.keys())
    states.remove('<S>')
    next_states = list(transition_prob.keys())
    next_states.remove('<S>')
    next_states.append('<E>')
    words = line.split(' ')

    track_back = []
    back_track_timestep = {}
    values = {}
    prev_state = '<S>'

    # Accumulate the initial state values
    for state in next_states:
        back_track_timestep[state] = prev_state
        emission = get_emission_prob(emission_prob, words[0], state)
        
        if emission == 0:
            if state in open_class:
                values[state] = transition_prob[prev_state][state]
            else:
                values[state] = 0
        elif emission == 1:
            values[state] = 0
        else:
            values[state] = transition_prob[prev_state][state] * emission_prob[words[0]][state]
    track_back.append(back_track_timestep)
    prev_values = values

    # Accumulate until T steps(Viterbi Algorithm)
    for word in words[1:]:
        values = {}
        back_track_timestep = {}
        for curr_state in next_states:
            emission = get_emission_prob(emission_prob, word, curr_state)

            if emission == 0:
                if curr_state not in open_class:
                    values[curr_state] = 0
                else:
                    max_value = 0
                    max_arg = 0
                    for prev_state in states:
                        value = prev_values[prev_state] * transition_prob[prev_state][curr_state]
                        if value > max_value:
                            max_value, max_arg = value, prev_state
                    values[curr_state] = max_value
                    back_track_timestep[curr_state] = max_arg

            elif emission == 1:
                values[curr_state] = 0

            else:
                #take max of prev states
                max_value = 0
                max_arg = 0
                for prev_state in states:
                    value = prev_values[prev_state] * transition_prob[prev_state][curr_state]
                    if value > max_value:
                        max_value, max_arg = value, prev_state
                values[curr_state] = max_value * emission_prob[word][curr_state]
                back_track_timestep[curr_state] = max_arg

        track_back.append(back_track_timestep)
        prev_values = values

    # Accumulate for the last step
    values = {}
    for state in states:
        values[state] = prev_values[state] * transition_prob[state]['<E>']
    final_state = arg_max(values)

    # Track back
    tags = []
    tags.insert(0, final_state)
    start = len(track_back)-1
    while final_state != '<S>':
        tags.insert(0, track_back[start][final_state])
        final_state = track_back[start][final_state]
        start -= 1
    tags.remove('<S>')

    return tags


def code_hmm(contents, transition_prob, emission_prob, open_class):

    lines = contents.split('\n')
    coded_data = ""
    for line in lines:
        if line != "":
            tags = hmmcode(line, transition_prob, emission_prob, open_class)
            # tags = baseline(line, transition_prob, emission_prob, open_class)
            words = line.split(' ')
            for index, word in enumerate(words):
                add_space = ' ' if index != len(words) - 1 else ''
                str = word + '/' + tags[index] + add_space
                coded_data += str
            coded_data += '\n'

    with open('hmmoutput.txt', 'w') as output:
        output.write(coded_data)


if __name__ == '__main__':

    file_path = sys.argv[1]
    # file_path = 'test.txt'
    with open(file_path, 'r') as readlike:
        contents = readlike.read()
    with open('hmmmodel.txt', 'r') as readlike:
        model_dict = json.load(readlike)

    transition_prob = model_dict['transition_prob']
    emission_prob = model_dict['emission_prob']
    open_class = model_dict['open_class']
    code_hmm(contents, transition_prob, emission_prob, open_class)

    # # Accumulate the initial state values
    # for state in next_states:
    #     back_track_timestep[state] = prev_state
    #     emission = get_emission_prob(emission_prob, words[0], state)
    #     if emission == 0:
    #
    #     else:
    #         values[state] = transition_prob[prev_state][state] * get_emission_prob(emission_prob, words[0], state)
    # track_back.append(back_track_timestep)
    # prev_values = values

    # values = {}
    # back_track_timestep = {}
    # for curr_state in next_states:
    #     max_value = 0
    #     #include emission probs
    #     max_arg = ""
    #     for prev_state in states:
    #         value = prev_values[prev_state] * transition_prob[prev_state][curr_state]
    #         if value > max_value:
    #             max_value, max_arg = value, prev_state
    #     values[curr_state] = max_value * get_emission_prob(emission_prob, word, curr_state)
    #     back_track_timestep[curr_state] = max_arg
    # track_back.append(back_track_timestep)
    # prev_values = values