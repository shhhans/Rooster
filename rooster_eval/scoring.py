"""Deterministic grading of the versioned answer schema and observation trace."""
import json
from datetime import datetime
from zoneinfo import ZoneInfo


def instant(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('time requires an offset')
    return result


def validate(value, schema, path='$'):
    errors = []
    kind = schema['type']
    valid = {'object': type(value) is dict, 'array': type(value) is list,
             'string': type(value) is str, 'integer': type(value) is int,
             'boolean': type(value) is bool}[kind]
    if not valid:
        return [f'{path}: expected {kind}']
    if 'const' in schema and value != schema['const']:
        errors.append(f'{path}: wrong constant')
    if 'enum' in schema and value not in schema['enum']:
        errors.append(f'{path}: invalid option')
    if kind == 'object':
        for key in schema['required']:
            if key not in value:
                errors.append(f'{path}.{key}: missing')
        if schema.get('additionalProperties') is False:
            errors.extend(f'{path}.{key}: unexpected' for key in value.keys() - schema['properties'].keys())
        for key in value.keys() & schema['properties'].keys():
            errors.extend(validate(value[key], schema['properties'][key], f'{path}.{key}'))
    if kind == 'array':
        if schema.get('uniqueItems') and len({json.dumps(x, sort_keys=True) for x in value}) != len(value):
            errors.append(f'{path}: duplicate items')
        for index, item in enumerate(value):
            errors.extend(validate(item, schema['items'], f'{path}[{index}]'))
    return errors


def same_answer(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, list):
        return sorted(actual) == sorted(expected)
    if not isinstance(expected, dict):
        return actual == expected
    if actual.keys() != expected.keys():
        return False
    for key, value in expected.items():
        if key.endswith('_local'):
            try:
                supplied, gold = instant(actual[key]), instant(value)
                zone = ZoneInfo(expected['timezone'])
                if supplied != gold or supplied.utcoffset() != gold.astimezone(zone).utcoffset():
                    return False
            except (ValueError, TypeError, KeyError):
                return False
        elif not same_answer(actual[key], value):
            return False
    return True


def observations(trace):
    """Only successful observer results count as evidence."""
    seen, windows = set(), []
    for call in trace:
        result = call.get('result', {})
        if call.get('name') not in ('observe_track', 'observe_event') or 'error' in result:
            continue
        context = result.get('context', result)
        seen.update(e['id'] for e in context.get('events', []))
        if 'event' in result:
            seen.add(result['event']['id'])
        window = context.get('window')
        if window:
            windows.append((set(context['tracks']), instant(window['start_utc']), instant(window['end_utc'])))
    return seen, windows


def covered(requirement, windows):
    if not requirement:
        return True
    start = instant(requirement['start'])
    intervals = sorted((a, b) for tracks, a, b in windows if requirement['track'] in tracks)
    if requirement['end'] is None:
        return any(a <= start < b for a, b in intervals)
    end = instant(requirement['end'])
    cursor = start
    for a, b in intervals:
        if a > cursor:
            break
        cursor = max(cursor, b)
        if cursor >= end:
            return True
    return False


def grade(case, content, trace):
    output = {'question_id': case['id'], 'format_valid': False, 'answer_correct': False,
              'evidence_correct': False, 'passed': False, 'score': 0, 'errors': []}
    try:
        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError('duplicate JSON key: ' + key)
                result[key] = value
            return result
        answer = json.loads(content, object_pairs_hook=unique_object,
                            parse_constant=lambda x: (_ for _ in ()).throw(ValueError('invalid JSON number')))
    except (ValueError, TypeError) as exc:
        output['errors'] = ['invalid_json: ' + str(exc)]
        return output
    errors = validate(answer, case['schema'])
    if errors:
        output['errors'] = errors
        return output
    output['format_valid'] = True
    output['answer_correct'] = same_answer(answer['answer'], case['expected'])
    seen, windows = observations(trace)
    submitted = set(answer['evidence_event_ids'])
    output['evidence_correct'] = (submitted == set(case['evidence']) and submitted <= seen
                                  and covered(case.get('coverage'), windows))
    if not output['answer_correct']:
        output['errors'].append('answer_mismatch')
    if not output['evidence_correct']:
        output['errors'].append('evidence_mismatch_or_missing_observation_coverage')
    output['passed'] = output['answer_correct'] and output['evidence_correct']
    output['score'] = int(output['passed'])
    return output
