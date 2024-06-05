from __future__ import annotations
from typing import Dict, Tuple
import copy
import random
import math
import numpy as np
import yaml

"""# Utils.py"""

##################### MACROURI #####################
INTERVALE = 'Intervale'
ZILE = 'Zile'
MATERII = 'Materii'
PROFESORI = 'Profesori'
SALI = 'Sali'
CAPACITATE = 'Capacitate'

def read_yaml_file(file_path: str) -> dict:
    '''
    Citeste un fișier yaml și returnează conținutul său sub formă de dicționar
    '''
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def acces_yaml_attributes(yaml_dict: dict):
    '''
    Primește un dicționar yaml și afișează datele referitoare la atributele sale
    '''

    print('Zilele din orar sunt:', yaml_dict[ZILE])
    print()
    print('Intervalele orarului sunt:', yaml_dict[INTERVALE])
    print()
    print('Materiile sunt:', yaml_dict[MATERII])
    print()
    print('Profesorii sunt:', end=' ')
    print(*list(yaml_dict[PROFESORI].keys()), sep=', ')
    print()
    print('Sălile sunt:', end=' ')
    print(*list(yaml_dict[SALI].keys()), sep=', ')


def get_profs_initials(profs: list) -> dict:
    '''
    Primește o listă de profesori

    Returnează două dicționare:
    - unul care are numele profesorilor drept chei și drept valori prescurtările lor (prof_to_initials[prof] = initiale)
    - unul care are prescurtările profesorilor drept chei și drept valori numele lor (initials_to_prof[initiale] = prof)
    '''

    initials_to_prof = {}
    prof_to_initials = {}
    initials_count = {}

    for prof in profs:
        name_components = prof.split(' ')
        initials = name_components[0][0] + name_components[1][0]

        if initials in initials_count:
            initials_count[initials] += 1
            initials += str(initials_count[initials])
        else:
            initials_count[initials] = 1

        initials_to_prof[initials] = prof
        prof_to_initials[prof] = initials

    return prof_to_initials, initials_to_prof


def allign_string_with_spaces(s: str, max_len: int, allignment_type: str = 'center') -> str:
    '''
    Primește un string și un număr întreg

    Returnează string-ul dat, completat cu spații până la lungimea dată
    '''

    len_str = len(s)

    if len_str >= max_len:
        raise ValueError(
            'Lungimea string-ului este mai mare decât lungimea maximă dată')

    if allignment_type == 'left':
        s = 6 * ' ' + s
        s += (max_len - len(s)) * ' '

    elif allignment_type == 'center':
        if len_str % 2 == 1:
            s = ' ' + s
        s = s.center(max_len, ' ')

    return s


def pretty_print_timetable_aux_zile(timetable: {str: {(int, int): {str: (str, str)}}}, input_path: str) -> str:
    '''
    Primește un dicționar ce are chei zilele, cu valori dicționare de intervale reprezentate ca tupluri de int-uri, cu valori dicționare de săli, cu valori tupluri (profesor, materie)

    Returnează un string formatat să arate asemenea unui tabel excel cu zilele pe linii, intervalele pe coloane și în intersecția acestora, ferestrele de 2 ore cu materiile alocate în fiecare sală fiecărui profesor
    '''

    max_len = 30

    profs = read_yaml_file(input_path)[PROFESORI].keys()
    profs_to_initials, _ = get_profs_initials(profs)

    table_str = '|           Interval           |             Luni             |             Marti            |           Miercuri           |              Joi             |            Vineri            |\n'

    no_classes = len(timetable['Luni'][(8, 10)])

    first_line_len = 187
    delim = '-' * first_line_len + '\n'
    table_str = table_str + delim

    for interval in timetable['Luni']:
        s_interval = '|'

        crt_str = allign_string_with_spaces(
            f'{interval[0]} - {interval[1]}', max_len, 'center')

        s_interval += crt_str

        for class_idx in range(no_classes):
            if class_idx != 0:
                s_interval += f'|{30 * " "}'

            for day in timetable:
                classes = timetable[day][interval]
                classroom = list(classes.keys())[class_idx]

                s_interval += '|'

                if not classes[classroom]:
                    s_interval += allign_string_with_spaces(
                        f'{classroom} - goala', max_len, 'left')
                else:
                    prof, subject = classes[classroom]
                    s_interval += allign_string_with_spaces(
                        f'{subject} : ({classroom} - {profs_to_initials[prof]})', max_len, 'left')

            s_interval += '|\n'
        table_str += s_interval + delim

    return table_str


def pretty_print_timetable_aux_intervale(timetable: {(int, int): {str: {str: (str, str)}}}, input_path: str) -> str:
    '''
    Primește un dicționar de intervale reprezentate ca tupluri de int-uri, cu valori dicționare de zile, cu valori dicționare de săli, cu valori tupluri (profesor, materie)

    Returnează un string formatat să arate asemenea unui tabel excel cu zilele pe linii, intervalele pe coloane și în intersecția acestora, ferestrele de 2 ore cu materiile alocate în fiecare sală fiecărui profesor
    '''

    max_len = 30

    profs = read_yaml_file(input_path)[PROFESORI].keys()
    profs_to_initials, _ = get_profs_initials(profs)

    table_str = '|           Interval           |             Luni             |             Marti            |           Miercuri           |              Joi             |            Vineri            |\n'

    no_classes = len(timetable[(8, 10)]['Luni'])

    first_line_len = 187
    delim = '-' * first_line_len + '\n'
    table_str = table_str + delim

    for interval in timetable:
        s_interval = '|' + \
            allign_string_with_spaces(
                f'{interval[0]} - {interval[1]}', max_len, 'center')

        for class_idx in range(no_classes):
            if class_idx != 0:
                s_interval += '|'

            for day in timetable[interval]:
                classes = timetable[interval][day]
                classroom = list(classes.keys())[class_idx]

                s_interval += '|'

                if not classes[classroom]:
                    s_interval += allign_string_with_spaces(
                        f'{classroom} - goala', max_len, 'left')
                else:
                    prof, subject = classes[classroom]
                    s_interval += allign_string_with_spaces(
                        f'{subject} : ({classroom} - {profs_to_initials[prof]})', max_len, 'left')

            s_interval += '|\n'
        table_str += s_interval + delim

    return table_str


def pretty_print_timetable(timetable: dict, input_path: str) -> str:
    '''
    Poate primi fie un dictionar de zile conținând dicționare de intervale conținând dicționare de săli cu tupluri (profesor, materie)
    fie un dictionar de intervale conținând dictionare de zile conținând dicționare de săli cu tupluri (profesor, materie)

    Pentru cazul în care o sală nu este ocupată la un moment de timp, se așteaptă 'None' în valoare, în loc de tuplu
    '''
    if 'Luni' in timetable:
        return pretty_print_timetable_aux_zile(timetable, input_path)
    else:
        return pretty_print_timetable_aux_intervale(timetable, input_path)

"""# check_constraints"""

#################### FUNCTII AUXILIARE ####################
def parse_interval(interval: str):
    '''
    Se parsează un interval de forma "Ora1 - Ora2" în cele 2 componente.
    '''

    intervals = interval.split('-')
    return int(intervals[0].strip()), int(intervals[1].strip())


def parse_subject_room_prof(subject_room_prof: str, nick_to_prof: dict):
    '''
    Se parsează un string de forma "Materie : Sala - Profesor" în cele 3 componente.
    '''

    if 'goala' in subject_room_prof:
        room = subject_room_prof.split('-')[0].strip()

        return None, room, None

    subject = subject_room_prof.split(':')[0].strip()
    room = subject_room_prof.split('(')[1].split('-')[0].strip()

    prof = subject_room_prof.split('-')[1][:-1].strip()
    prof = nick_to_prof[prof]
    return subject, room, prof


def get_timetable(timetable_specs: dict, output_name: str, debug_flag: bool = False):
    '''
    Pe baza specificațiilor din fișierul de intrare, se reprezintă intern orarul din fișierul de ieșire.
    '''
    timetable = {day: {eval(interval): {
    } for interval in timetable_specs[INTERVALE]} for day in timetable_specs[ZILE]}

    _, initials_to_prof = get_profs_initials(timetable_specs[PROFESORI])

    if debug_flag:
        print(initials_to_prof)
        print()

    interval = None

    with open(output_name, 'r') as file:
        for line in file:
            if line[0] != '|':
                continue

            crt_parsing = line.strip().split('|')
            crt_parsing = [x.strip() for x in crt_parsing]
            if not crt_parsing:
                continue

            if crt_parsing[1] == 'Interval':
                continue

            crt_interval = crt_parsing[1]

            if crt_interval != '':
                interval = parse_interval(crt_interval)
            # print(parse_subject_room_prof(crt_parsing[2], timetable_specs[PROFESORI]))

            idx = 2

            for day in timetable_specs[ZILE]:
                subject, room, prof = parse_subject_room_prof(
                    crt_parsing[idx], initials_to_prof)
                if subject:
                    # ACEEASI SALA ESTE OCUPATA DE 2 MATERII IN ACELASI INTERVAL
                    if room in timetable[day][interval]:
                        print(
                            f'Sala {room} este ocupata de 2 materii in acelasi interval!')
                        raise Exception(
                            'Sala ocupata de 2 materii in acelasi interval!')

                    timetable[day][interval][room] = prof, subject
                else:
                    timetable[day][interval][room] = None
                idx += 1
    print(timetable)
    return timetable


def check_mandatory_constraints(timetable: {str: {(int, int): {str: (str, str)}}}, timetable_specs: dict):
    '''
    Se verifică dacă orarul generat respectă cerințele obligatorii pentru a fi un orar valid.
    '''

    constrangeri_incalcate = 0

    acoperire_target = timetable_specs[MATERII]

    acoperire_reala = {subject: 0 for subject in acoperire_target}

    ore_profesori = {prof: 0 for prof in timetable_specs[PROFESORI]}

    for day in timetable:
        for interval in timetable[day]:
            profs_in_crt_interval = []
            for room in timetable[day][interval]:
                if timetable[day][interval][room]:
                    prof, subject = timetable[day][interval][room]
                    acoperire_reala[subject] += timetable_specs[SALI][room][CAPACITATE]

                    # PROFESORUL PREDĂ 2 MATERII ÎN ACELAȘI INTERVAL
                    if prof in profs_in_crt_interval:
                        print(
                            f'Profesorul {prof} preda 2 materii in acelasi interval!')
                        constrangeri_incalcate += 1
                    else:
                        profs_in_crt_interval.append(prof)

                    # MATERIA NU SE PREDA IN SALA
                    if subject not in timetable_specs[SALI][room][MATERII]:
                        print(f'Materia {subject} nu se preda în sala {room}!')
                        constrangeri_incalcate += 1

                    # PROFESORUL NU PREDA MATERIA
                    if subject not in timetable_specs[PROFESORI][prof][MATERII]:
                        print(
                            f'Profesorul {prof} nu poate preda materia {subject}!')
                        constrangeri_incalcate += 1

                    ore_profesori[prof] += 1

    # CONDITIA DE ACOPERIRE
    for subject in acoperire_target:
        if acoperire_reala[subject] < acoperire_target[subject]:
            print(f'Materia {subject} nu are acoperirea necesară!')
            constrangeri_incalcate += 1

    # CONDITIA DE MAXIM 7 ORE PE SĂPTĂMÂNĂ
    for prof in ore_profesori:
        if ore_profesori[prof] > 7:
            print(f'Profesorul {prof} tine mai mult de 7 sloturi!')
            constrangeri_incalcate += 1

    return constrangeri_incalcate


def check_optional_constraints(timetable: {str: {(int, int): {str: (str, str)}}}, timetable_specs: dict):
    '''
    Se verifică dacă orarul generat respectă cerințele profesorilor pentru a fi un orar valid.
    '''

    constrangeri_incalcate = 0

    for prof in timetable_specs[PROFESORI]:
        for const in timetable_specs[PROFESORI][prof]['Constrangeri']:
            if const[0] != '!':
                continue
            else:
                const = const[1:]

                if const in timetable_specs[ZILE]:
                    day = const
                    if day in timetable:
                        for interval in timetable[day]:
                            for room in timetable[day][interval]:
                                if timetable[day][interval][room]:
                                    crt_prof, _ = timetable[day][interval][room]
                                    if prof == crt_prof:
                                        print(
                                            f'Profesorul {prof} nu dorește să predea în ziua {day}!')
                                        constrangeri_incalcate += 1

                elif '-' in const:
                    interval = parse_interval(const)
                    start, end = interval

                    if start != end - 2:
                        intervals = [(i, i + 2) for i in range(start, end, 2)]
                    else:
                        intervals = [(start, end)]

                    for day in timetable:
                        for interval in intervals:
                            if interval in timetable[day]:
                                for room in timetable[day][interval]:
                                    if timetable[day][interval][room]:
                                        crt_prof, _ = timetable[day][interval][room]
                                        if prof == crt_prof:
                                            print(
                                                f'Profesorul {prof} nu dorește să predea în intervalul {interval}!')
                                            constrangeri_incalcate += 1

    return constrangeri_incalcate

"""## Utilitati"""

def has_negative_preference(professor, day, interval, timetable_specs):
    for constraint in timetable_specs['Profesori'][professor]['Constrangeri']:
        if constraint.startswith('!'):
            constraint = constraint[1:]
            if constraint == day or (constraint.startswith('-') and parse_interval(constraint) == interval):
                return True
    return False

def parse_interval_alt(interval_str: str):
    stripped = interval_str.strip('()')
    start, end = stripped.split(',')
    return (int(start.strip()), int(end.strip()))

"""# Model Baza + SA HC"""

def generate_schedule(timetable_specs, seed: int):
    random.seed(seed)
    days = timetable_specs['Zile']
    raw_intervals = list(timetable_specs['Intervale'])
    intervals = [parse_interval_alt(interval) for interval in raw_intervals]
    rooms = timetable_specs['Sali']
    professors = timetable_specs['Profesori']
    subjects = timetable_specs['Materii']

    # Precompute valid subjects for each room and professors for each subject
    room_subject_map = {room: [subj for subj in rooms[room] if subj in subjects] for room in rooms}
    subject_professor_map = {subject: [prof for prof, details in professors.items() if subject in details['Materii']] for subject in subjects}

    # Initialize the timetable
    timetable = {day: {interval: {} for interval in intervals} for day in days}

    # Attempt to fill the timetable, prioritizing hard constraints and then soft constraints
    for day in timetable:
        for interval in timetable[day]:
            for room in rooms:
                if random.choice([True, False]):  # Random assignment decision
                    suitable_subjects = room_subject_map[room]
                    if suitable_subjects:
                        # Prioritize subjects by the number of available professors (fewer is better to reduce conflicts)
                        suitable_subjects.sort(key=lambda subj: len(subject_professor_map[subj]))
                        for chosen_subject in suitable_subjects:
                            suitable_professors = subject_professor_map[chosen_subject]
                            # Shuffle to avoid bias
                            random.shuffle(suitable_professors)
                            for chosen_professor in suitable_professors:
                                # Check for conflict with professor's existing assignments
                                if not any(timetable[day][i].get(r) == (chosen_professor, chosen_subject)
                                           for i in intervals for r in rooms):
                                    # Check for professor's negative preferences
                                    if not has_negative_preference(chosen_professor, day, interval, timetable_specs):
                                        # Assign if no conflicts and no negative preferences
                                        timetable[day][interval][room] = (chosen_professor, chosen_subject)
                                        break
                            if room in timetable[day][interval]:
                                break  # Stop if we made an assignment
                else:
                    timetable[day][interval][room] = None

    return timetable

"""## Functii Cost"""

def compute_conflicts(timetable_specs, timetable):
    constrangeri_incalcate = 0
    acoperire_target = timetable_specs['Materii']
    acoperire_reala = {subject: 0 for subject in acoperire_target}
    ore_profesori = {prof: 0 for prof in timetable_specs['Profesori']}
    for day in timetable:
        for interval in timetable[day]:
            profs_in_crt_interval = []
            for room in timetable[day][interval]:
                if timetable[day][interval][room]:
                    prof, subject = timetable[day][interval][room]
                    acoperire_reala[subject] += timetable_specs['Sali'][room]['Capacitate']
                    if prof in profs_in_crt_interval:
                        constrangeri_incalcate += 1
                    else:
                        profs_in_crt_interval.append(prof)
                    if subject not in timetable_specs['Sali'][room]['Materii']:
                        constrangeri_incalcate += 1
                    if subject not in timetable_specs['Profesori'][prof]['Materii']:
                        constrangeri_incalcate += 1
                    ore_profesori[prof] += 1
    for subject in acoperire_target:
        if acoperire_reala[subject] < acoperire_target[subject]:
            constrangeri_incalcate += 1
    for prof in ore_profesori:
        if ore_profesori[prof] > 7:
            constrangeri_incalcate += 1
    return constrangeri_incalcate * 100

def compute_soft_conflicts(timetable_specs, timetable):
    constrangeri_incalcate = 0
    for prof in timetable_specs['Profesori']:
        for const in timetable_specs['Profesori'][prof]['Constrangeri']:
            if const[0] != '!':
                continue
            else:
                const = const[1:]
                if const in timetable_specs['Zile']:
                    day = const
                    if day in timetable:
                        for interval in timetable[day]:
                            for room in timetable[day][interval]:
                                if timetable[day][interval][room]:
                                    crt_prof, _ = timetable[day][interval][room]
                                    if prof == crt_prof:
                                        constrangeri_incalcate += 1
                elif '-' in const:
                    interval = parse_interval(const)
                    start, end = interval
                    if start != end - 2:
                        intervals = [(i, i + 2) for i in range(start, end, 2)]
                    else:
                        intervals = [(start, end)]
                    for day in timetable:
                        for interval in intervals:
                            if interval in timetable[day]:
                                for room in timetable[day][interval]:
                                    if timetable[day][interval][room]:
                                        crt_prof, _ = timetable[day][interval][room]
                                        if prof == crt_prof:
                                            constrangeri_incalcate += 1
    return constrangeri_incalcate

class Schedule:
    def __init__(
            self,
            timetable_specs,
            timetable = None,
            conflicts = None,
            seed=42,
    ) -> None:
        self.timetable_specs = timetable_specs
        self.timetable = timetable if timetable else generate_schedule(
            timetable_specs, seed)
        self.nconflicts = conflicts if conflicts is not None else compute_conflicts(
            self.timetable_specs, self.timetable) + compute_soft_conflicts(self.timetable_specs, self.timetable)
        np.random.seed(seed)

    def conflicts(self):
        # Note: This is a costly operation, so we cache the result
        return self.nconflicts if self.nconflicts is not None else compute_conflicts(self.timetable_specs, self.timetable) + compute_soft_conflicts(self.timetable_specs, self.timetable)

    def get_next_states(self):
      neighbors = []
      days = list(self.timetable.keys())
      raw_intervals = list(self.timetable_specs['Intervale'])
      intervals = [parse_interval_alt(interval) for interval in raw_intervals]  # '(8,10)' => (8,10)
      rooms = list(self.timetable_specs['Sali'].keys())
      professors = list(self.timetable_specs['Profesori'].keys())
      subjects = list(self.timetable_specs['Materii'].keys())

      num_neighbors = 5  # Number of neighbors to generate

      for _ in range(num_neighbors):
          neighbor = self.create_neighbor(days, intervals, rooms, professors, subjects)
          neighbors.append(neighbor)

      return neighbors

    def create_neighbor(self, days, intervals, rooms, professors, subjects):
        new_timetable = copy.deepcopy(self.timetable)
        day = random.choice(days)
        interval = random.choice(intervals)
        room = random.choice(rooms)

        if random.choice([True, False]):  # 50% chance to swap
            other_day = random.choice(days)
            other_interval = random.choice(intervals)
            other_room = random.choice(rooms)

            # Swap the assignments if both are non-empty
            if new_timetable[day][interval].get(room) and new_timetable[other_day][other_interval].get(other_room):
                new_timetable[day][interval][room], new_timetable[other_day][other_interval][other_room] = \
                    new_timetable[other_day][other_interval][other_room], new_timetable[day][interval][room]
        else:
            potential_subjects = [
                subj for subj in subjects if subj in self.timetable_specs['Sali'][room]['Materii']]
            if potential_subjects:
                new_subject = random.choice(potential_subjects)
                # Only assign a teacher that can teach the subject
                potential_professors = [
                    prof for prof in professors if new_subject in self.timetable_specs['Profesori'][prof]['Materii']]
                if potential_professors:
                    # Randomly select a professor
                    new_professor = random.choice(potential_professors)
                    new_timetable[day][interval][room] = (
                        new_professor, new_subject)

        return Schedule(self.timetable_specs, new_timetable)

    def optimize_schedule_with_statistics(self, max_iterations=30_000, restarts=3, initial_temp=3000, cooling_rate=0.3):
        best_schedule = None
        best_conflicts = float('inf')
        conflict_history = []

        for _ in range(restarts):
            # Generate a new random initial schedule for each restart
            current_schedule = Schedule(self.timetable_specs)
            current_conflicts = current_schedule.conflicts()
            temperature = initial_temp  # Reset the temperature for simulated annealing
            iteration_conflicts = [current_conflicts]

            for _ in range(max_iterations):
                temperature *= cooling_rate  # Gradually decrease the temperature
                neighbors = current_schedule.get_next_states()
                if not neighbors:
                    continue  # If no neighbors are generated, skip to the next iteration TODO: Maybe we should always generate neighbors? or default to None?

                next_schedule = random.choice(neighbors)
                next_conflicts = next_schedule.conflicts()

                # Record conflicts before checking conditions
                iteration_conflicts.append(next_conflicts)

                #  Simulated annealing acceptance criterion (accept with probability exp(-deltaE / T)) where deltaE = next_conflicts - current_conflicts
                #  And T is the current temperature (higher temperature -> more likely to accept worse solutions)
                #  we do + 0.1 to avoid division by zero
                if next_conflicts < current_conflicts or math.exp((current_conflicts - next_conflicts) / (temperature + 0.1)) > random.random():
                    current_schedule = next_schedule
                    current_conflicts = next_conflicts

                if current_conflicts < best_conflicts:
                    best_schedule = current_schedule
                    best_conflicts = current_conflicts

                if best_conflicts == 0:
                    break

            conflict_history.append(iteration_conflicts)

            if best_conflicts == 0:
                break

        print(f"Best schedule found has {best_conflicts} conflicts.")
        return best_schedule, conflict_history

    def __str__(self):
        return "Schedule: {} conflicts".format(self.nconflicts)

    def get_timetable(self):
        return self.timetable

    def clone(self):
        return Schedule(self.timetable_specs, copy.deepcopy(self.timetable), self.nconflicts)




"""## Model 1 Test Case Dummy"""
print("test 1 Dummy ")
timetable_specs = read_yaml_file('dummy.yaml')
schedule = Schedule(timetable_specs)

optimized_schedule, history = schedule.optimize_schedule_with_statistics()

print(check_mandatory_constraints(optimized_schedule.timetable,optimized_schedule.timetable_specs))
print(check_optional_constraints(optimized_schedule.timetable, optimized_schedule.timetable_specs))

"""## Test Case 2"""
print("test 2 orar_mic_exact ")
timetable_specs = read_yaml_file('orar_mic_exact.yaml')
schedule = Schedule(timetable_specs)

optimized_schedule, history = schedule.optimize_schedule_with_statistics()

print(check_mandatory_constraints(optimized_schedule.timetable,optimized_schedule.timetable_specs))
print(check_optional_constraints(optimized_schedule.timetable, optimized_schedule.timetable_specs))

"""## Test Case 3"""
print("test 3 orar_mediu_relaxat ")
timetable_specs = read_yaml_file('orar_mediu_relaxat.yaml')

optimized_schedule, history = schedule.optimize_schedule_with_statistics()

print(check_mandatory_constraints(optimized_schedule.timetable,optimized_schedule.timetable_specs))
print(check_optional_constraints(optimized_schedule.timetable, optimized_schedule.timetable_specs))

"""## Test Case 4"""
print("test 4 orar_mare_relaxat ")
timetable_specs = read_yaml_file('orar_mare_relaxat.yaml')
schedule = Schedule(timetable_specs)

optimized_schedule, history = schedule.optimize_schedule_with_statistics()

print(check_mandatory_constraints(optimized_schedule.timetable,optimized_schedule.timetable_specs))
print(check_optional_constraints(optimized_schedule.timetable, optimized_schedule.timetable_specs))

"""## Test Case 5"""
print("test 5 orar_constrans_incalcat ")
timetable_specs = read_yaml_file('orar_constrans_incalcat.yaml')
schedule = Schedule(timetable_specs)

optimized_schedule, history = schedule.optimize_schedule_with_statistics()

print(check_mandatory_constraints(optimized_schedule.timetable,optimized_schedule.timetable_specs))
print(check_optional_constraints(optimized_schedule.timetable, optimized_schedule.timetable_specs))
