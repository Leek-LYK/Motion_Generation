"""User-authorized M5v3 difficulty scope, based only on prompt metadata.

This module never reads generated scores or changes motion. Simple raise/lower,
walk/stop and ordinary turns remain basic even when the prompt contains 'then'.
"""
import re

VERSION = 'm5-basic-motion-scope-v1'
BASIC_GROUPS = ('walk', 'jog', 'raise_arms', 'turn')
RULES = {
    'jump_or_flight': r'\b(?:jump(?:s|ed|ing)?|hop(?:s|ped|ping)?|leap(?:s|ed|ing)?|airborne)\b|跳跃|起跳|蹦跳',
    'low_posture': r'\b(?:squat(?:s|ted|ting)?|crouch(?:es|ed|ing)?|crawl(?:s|ed|ing)?|kneel(?:s|ed|ing)?|all fours|lying|prone|supine|deeply bent)\b|低姿态|深蹲|下蹲|匍匐|爬行|跪姿',
    'dance_or_acrobatics': r'\b(?:danc(?:e|es|ed|ing)|ballet|breakdanc\w*|hip[- ]hop|dab(?:bing)?|cartwheel(?:s|ing)?|somersault(?:s|ing)?|acrobatic\w*|kung[- ]fu|martial arts)\b|舞蹈|街舞|武术|空翻',
    'large_rotation': r'\b(?:360(?:\s*(?:deg(?:ree)?s?))?|full\s+(?:(?:counterclockwise|clockwise)\s+)?(?:turn|circle|rotation)|spin(?:s|ning)?|twirl(?:s|ing)?)\b|完整转圈|连续旋转',
}
FAST = r'\b(?:fast(?:er|est)?|quick(?:ly|er)?|rapid(?:ly)?|sprint(?:s|ed|ing)?|dash(?:es|ed|ing)?|brisk(?:ly)?|swift(?:ly)?)\b|快速|快跑|冲刺|疾跑'
LOCOMOTION_OR_TURN = ('walk', 'jog', 'turn')


def classify(entry):
    text = entry.get('text', entry.get('prompt', ''))
    group = entry['group']; reasons = [name for name, pattern in RULES.items() if re.search(pattern, text, re.I)]
    if group == 'jump' and 'jump_or_flight' not in reasons: reasons.append('jump_or_flight')
    if group == 'squat' and 'low_posture' not in reasons: reasons.append('low_posture')
    if group in LOCOMOTION_OR_TURN and re.search(FAST, text, re.I): reasons.append('fast_locomotion_or_turn')
    if group not in BASIC_GROUPS and not reasons: reasons.append('outside_basic_groups')
    return {'difficulty': 'extended' if reasons else 'basic', 'group': group,
            'reasons': sorted(set(reasons)), 'classification_inputs': ['text', 'group']}
