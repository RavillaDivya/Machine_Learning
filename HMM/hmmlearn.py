import json
import sys

def count(contents):

    emission_count = {}
    transition_count = {}
    open_class = {}
    tag_count = {}
    start_count = 0
    lines = contents.split('\n')
    for line in lines:
        if line != "":
            prev_state = '<S>'
            start_count += 1
            tagged_words_list = line.split(' ')
            for index, tagged_word in enumerate(tagged_words_list):
                cur_state = tagged_word.split('/')[-1]
                word = tagged_word[:-(len(cur_state) + 1)]

                # Add to emission_count
                if word in emission_count.keys():
                    if cur_state in emission_count[word]:
                        emission_count[word][cur_state] += 1
                    else:
                        emission_count[word][cur_state] = 1
                        # Adding the association to Open_class dictionary
                        if cur_state in open_class.keys():
                            open_class[cur_state] += 1
                        else:
                            open_class[cur_state] = 1

                else:
                    emission_count[word] = {}
                    emission_count[word][cur_state] = 1
                    # Adding the association to Open_class dictionary
                    if cur_state in open_class.keys():
                        open_class[cur_state] += 1
                    else:
                        open_class[cur_state] = 1

                # Add state to all_state_dictionary
                if cur_state in tag_count.keys():
                    tag_count[cur_state] += 1
                else:
                    tag_count[cur_state] = 1

                # Add to transition_count
                if prev_state in transition_count.keys():
                    if cur_state in transition_count[prev_state]:
                        transition_count[prev_state][cur_state] += 1
                    else:
                        transition_count[prev_state][cur_state] = 1
                else:
                    transition_count[prev_state] = {}
                    transition_count[prev_state][cur_state] = 1

                prev_state = cur_state

            # Add end tag
            cur_state = '<E>'
            if prev_state in transition_count.keys():
                if cur_state in transition_count[prev_state]:
                    transition_count[prev_state][cur_state] += 1
                else:
                    transition_count[prev_state][cur_state] = 1
            else:
                transition_count[prev_state] = {}
                transition_count[prev_state][cur_state] = 1

    # print(transition_count)
    # print(emission_count)
    # print(tag_count)
    emission_prob = {}
    transition_prob = {}

    # Calculate Emission Probabilities
    for word in emission_count.keys():
        temp_dict = {}
        for state in emission_count[word].keys():
            temp_dict[state] = emission_count[word][state] / tag_count[state]
        emission_prob[word] = temp_dict

    prev_states = list(transition_count.keys())
    end_states = list(tag_count.keys())
    end_states.append('<E>')
    tag_count['<S>'] = start_count

    # Calculate Transition Probabilities
    for prev_state in prev_states:
        temp_dict = {}
        for curr_state in end_states:
            if curr_state in transition_count[prev_state].keys():
                temp_dict[curr_state] = (transition_count[prev_state][curr_state] + 1) / (
                        tag_count[prev_state] + len(end_states))
            else:
                temp_dict[curr_state] = 1 / (tag_count[prev_state] + len(end_states))
        transition_prob[prev_state] = temp_dict

    # Open Class(Top 5)
    open_class = sorted(open_class, key=open_class.get, reverse=True)[:5]

    return transition_prob, emission_prob, open_class
    

if __name__ == '__main__':

    file_path = sys.argv[1]
    # file_path = 'hmm-training-data/it_isdt_train_tagged.txt'
    with open(file_path) as infile:
        contents = infile.read()
    transition_prob, emission_prob, open_class = count(contents)
    hmm_model = {'transition_prob': transition_prob, 'emission_prob': emission_prob, 'open_class': open_class}
    with open('hmmmodel.txt', 'w') as write_file:
        write_file.write(json.dumps(hmm_model, indent=4))