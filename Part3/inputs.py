"""
P: passable
I: impassable
NW: No wind
WU: wind up
WD: wind down
WL: wind left
WR: wind right
"""
inputs_list = [
    # 0 simple test case
    dict(map=(('P_NW', 'P_NW',),
              ('P_NW', 'P_NW',)),
         drone_location=(0, 0),
         packages=[('A', (0, 1))],
         target_location=(1, 1),
         success_rate=1.),
    # 1 Introducing wind and lower action success rate
    dict(map=(('P_NW', 'P_NW',),
              ('P_WD', 'P_WD',),
              ('P_NW', 'P_NW',)),
         drone_location=(0, 0),
         packages=[('A', (0, 1))],
         target_location=(2, 0),
         success_rate=.9),
    # 2 Introducing random drone initial positions
    dict(map=(('P_NW', 'P_NW',),
              ('P_WD', 'P_WD',),
              ('P_NW', 'P_NW',)),
         drone_location='random',
         packages=[('A', (0, 1))],
         target_location=(2, 0),
         success_rate=.9),
    # 3 Introducing larger map and multiple packages
    dict(map=(('P_NW', 'P_NW', 'P_WU', 'P_WU', 'P_NW'),
              ('P_NW', 'P_NW', 'P_WU', 'P_WU', 'P_NW'),
              ('P_NW', 'P_NW', 'P_WU', 'P_WU', 'P_NW'),
              ('P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW'),),
         drone_location=(0, 0),
         packages=[('A', (3, 3)), ('B', (3, 2))],
         target_location=(3, 4),
         success_rate=.9),
    # 4 Introducing walls
    dict(map=(('P_NW', 'I_NW', 'P_WU', 'P_WU', 'P_NW'),
              ('P_NW', 'I_NW', 'P_WU', 'I_WU', 'P_NW'),
              ('P_NW', 'I_NW', 'P_WU', 'I_WU', 'P_NW'),
              ('P_NW', 'P_NW', 'P_WU', 'I_WU', 'P_NW'),),
         drone_location=(0, 0),
         packages=[('A', (1, 0))],
         target_location=(3, 4),
         success_rate=1.),
    # 5 Introducing larger map and multiple packages
    dict(map=(('P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW'),
              ('P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW'),
              ('P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW'),
              ('P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW'),
              ('P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW'),
              ('P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW'),
              ('P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW'),
              ('P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW'),
              ('P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW')),
         drone_location='random',
         packages=[('A', (1, 3)), ('B', (3, 8)), ('C', (4, 0)), ('D', (8, 7)), ('E', (6, 5))],
         target_location=(0, 0),
         success_rate=.9),
    # 6 Like 5 but has winds and walls
    dict(map=(('P_NW', 'P_NW', 'I_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW'),
              ('P_WD', 'P_WU', 'I_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW'),
              ('P_WD', 'P_WU', 'I_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW'),
              ('P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'I_NW', 'P_NW'),
              ('P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'I_NW', 'P_NW'),
              ('P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'I_NW', 'P_NW'),
              ('P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW'),
              ('P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'I_NW', 'P_NW', 'P_NW', 'P_NW'),
              ('P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'I_NW', 'P_NW', 'P_NW', 'P_NW')),
         drone_location=(0, 0),
         packages=[('A', (1, 3)), ('B', (3, 8)), ('C', (4, 0)), ('D', (8, 7)), ('E', (6, 5))],
         target_location=(0, 0),
         success_rate=1.),
    # 7 windy hallways
    dict(map=(('P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW'),
              ('P_WU', 'P_WD', 'P_WD', 'P_WD', 'P_WD', 'P_WD', 'P_WD', 'P_WD', 'P_WU'),
              ('P_WU', 'P_WD', 'P_WD', 'P_WD', 'P_WD', 'P_WD', 'P_WD', 'P_WD', 'P_WU'),
              ('P_WU', 'P_WD', 'P_WD', 'P_WD', 'P_WD', 'P_WD', 'P_WD', 'P_WD', 'P_WU'),
              ('P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW', 'P_NW')),
         drone_location=(0, 0),
         packages=[('A', (0, 2)), ('B', (0, 3)), ('C', (0, 4)), ('D', (0, 5)), ('E', (0, 6))],
         target_location=(4, 4),
         success_rate=1.),
]
