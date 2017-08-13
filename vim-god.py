
import unittest

SPACE_CHARS = set(['\n', '\t', ' '])

# CharSet {%{

class AbstractPossibility(object):

    def canContain(self, chars, variables={}):
        raise NotImplementedError

    def __contains__(self, char):
        return self.canContain({char})
    def __le__(self, other):
        raise NotImplementedError

    def mustContain(self, chars):
        raise NotImplementedError

    def add(self, chars):
        raise NotImplementedError
    def mustBeIn(self, chars):
        raise NotImplementedError
    def mustNotBeIn(self, chars):
        raise NotImplementedError

    def isEmpty(self):
        raise NotImplementedError

    def fixVariables(self):
        return set({})

    def internal_state(self):
        return (type(self),)
    def __eq__(self, other):
        return self.internal_state() == other.internal_state()
    def __hash__(self):
        return hash(self.internal_state())

class RemovePossibilitiy(AbstractPossibility):

    def __init__(self, removeChars, possibility):
        self._removeChars = removeChars
        self._possibility = possibility

    def mustContain(self, chars):
        chars_left = chars | self._removeChars
        return self._possibility.mustContain(chars_left)

    def canContain(self, chars, variables={}):
        chars_left = set(chars) - self._removeChars
        if not chars_left:
            return False
        return self._possibility.canContain(chars_left, variables)

    def add(self, chars):
        removeChars = set(self._removeChars) - chars
        return RemovePossibilitiy(removeChars, self._possibility.add(chars))

    def mustBeIn(self, chars):
        possibleChars = chars - self._removeChars
        newPossibility = self._possibility.mustBeIn(chars)
        return RemovePossibilitiy(possibleChars, newPossibility)

    def mustNotBeIn(self, chars):
        removeChars = chars | self._removeChars
        return RemovePossibilitiy(removeChars, self._possibility)

    def isEmpty(self):
        return self._possibility.isLimitedTo(self._removeChars)

    def fixVariables(self):
        return self._possibility.fixVariables()

    def internal_state(self):
        state = super(RemovePossibilitiy, self).internal_state()
        rc = list(self._removeChars)
        rc.sort()
        return (state, tuple(rc), self._possibility.internal_state())

    def __str__(self):
        rc = list(self._removeChars)
        rc.sort()
        return '{%s \\ %s}' % (self._possibility, rc)

    def __le__(self, other):
        return self._possibility <= other.add(self._removeChars)

class AddPossibility(AbstractPossibility):

    def canContain(self, chars, variables={}):
        if self.getStateId() in variables:
            return variables[self.getStateId()] in chars
        else:
            return True

    def isEmpty(self):
        return False

    def isLimitedTo(self, chars):
        raise NotImplementedError

    def getStateId(self):
        raise NotImplementedError

class XCharsPossibilities(AddPossibility):

    def __init__(self, stateId, chars):
        self._stateId = stateId
        self._possibilities = set()
        for char in chars:
            self._possibilities.add(char)
        assert self._possibilities

    def canContain(self, chars, variables={}):
        if not bool(chars & self._possibilities):
            return False
        return super(XCharsPossibilities, self).canContain(chars, variables)

    def mustContain(self, chars):
        return not bool(self._possibilities - chars)

    def add(self, chars):
        return XCharsPossibilities(self._stateId, self._possibilities | chars)

    def mustBeIn(self, chars):
        assert chars & self._possibilities
        return XCharsPossibilities(self._stateId, chars & self._possibilities)

    def mustNotBeIn(self, chars):
        assert not self.mustContain(chars)
        return XCharsPossibilities(self._stateId, self._possibilities - chars)

    def getStateId(self):
        return self._stateId

    def isLimitedTo(self, chars):
        return not (self._possibility - chars)

    def fixVariables(self):
        return {self._stateId}

    def internal_state(self):
        state = super(XCharsPossibilities, self).internal_state()
        tp = tuple(self._possibilities)
        return (state, self._stateId, tp)

    def __str__(self):
        return str(self._possibilities)

    def __le__(self, other):
        for char in self._possibilities:
            if char not in other:
                return False
        return True

class OneCharPossibility(XCharsPossibilities):

    def __init__(self, stateId, char):
        assert '\r' != char
        super(OneCharPossibility, self).__init__(stateId, {char})

class AnyCharPossibility(AddPossibility):

    def __init__(self, stateId):
        self._stateId = stateId

    def mustContain(self, chars):
        return False

    def __le__(self, other):
        return isinstance(other, AnyCharPossibility)

    def add(self, chars):
        return self

    def mustBeIn(self, chars):
        return XCharsPossibilities(self._stateId, chars)

    def mustNotBeIn(self, chars):
        return RemovePossibilitiy(chars, self)

    def getStateId(self):
        return self._stateId

    def isLimitedTo(self, chars):
        #TODO: Ensure that all the chars exist (...)
        return False

    def fixVariables(self):
        return {self._stateId}

    def internal_state(self):
        state = super(AnyCharPossibility, self).internal_state()
        return (state, self._stateId)

    def __str__(self):
        return '{...}'

# }%}

# EditorState {%{
class EditorState(object):
    MODE_INSERT = 0
    MODE_NORMAL = 1

    def __init__(self, content=''):
        self._position = 0
        self._mode = EditorState.MODE_NORMAL
        # a way to prevent a char from appearing in another one?
        self._idState = 0

        self._content = []
        for char in content:
            self._content.append(OneCharPossibility(self.nextState(), char))

    def internal_state(self):
        content_tuple = tuple(self._content)
        return (self._position, self._mode, self._idState, content_tuple )

    def __eq__(self, otherState):
        return self.internal_state() == otherState.internal_state()

    def __hash__(self):
        return hash(self.internal_state())

    @staticmethod
    def copy(other_state):
        other = EditorState()
        other._content = other_state._content.copy()
        other._mode = other_state._mode
        other._position = other_state._position
        other._idState = other_state._idState
        return other

    def nextState(self):
        val = self._idState
        self._idState += 1
        return val

    def force(self, position, chars):
        # we cannot force a char which is not valid
        assert self._content[position].canContain(chars)

        newstate = EditorState.copy(self)
        newstate._content[position] = newstate._content[position].mustBeIn(chars)
        return newstate

    def forceNot(self, position, chars):
        newPossibility = self._content[position].mustNotBeIn(chars)

        #TODO: It might be empty
        assert not newPossibility.isEmpty()

        newstate = EditorState.copy(self)
        newstate._content[position] = newPossibility

        return newstate

    def remove(self, position):
        newstate = EditorState.copy(self)
        del newstate._content[position]
        return newstate

    def insert(self, position, possibility):
        newstate = EditorState.copy(self)
        newstate._content.insert(position, possibility)
        newstate._position += 1
        return newstate

    def chars(self, pos):
        return self._content[pos]

    def canMatch(self, text):

        var_values = {}

        for a1, a2 in zip(self._content, text):
            var_ids = a1.fixVariables()

            for var_id in var_ids:
                if var_id in var_values and a2 != var_values[var_id]:
                    return False
                var_values[var_id] = a2

        for a1, a2 in zip(self._content, text):
            if not a1.canContain({a2}, var_values):
                return False

        return len(text) == len(self._content)

    def __str__(self):
        ret = ['(']
        ret.append({EditorState.MODE_INSERT: 'I', EditorState.MODE_NORMAL: 'E'}.get(self._mode))
        ret.append(',')
        ret.append(self._position)
        ret.append(', [')
        for elem in self._content:
            ret.append(elem)
        ret.append(']')

        return ''.join(map(str,ret))

    def lastPos(self):
        return len(self._content)

    def position(self):
        # char inserted before position in insert mode
        # char selected in normal mode
        return self._position

    def setPosition(self, position):
        if self.lastPos() == 0:
            assert position == 0
        elif self._mode == EditorState.MODE_NORMAL:
            assert position < self.lastPos() or \
                    self.chars(self.lastPos()-1).mustContain({'\n'})
        else: # EditorState.MODE_INSERT
            assert position < self.lastPos() + 1

        newstate = EditorState.copy(self)
        newstate._position = position
        return newstate

    def mode(self):
        return self._mode

    def setMode(self, new_mode):
        newstate = EditorState.copy(self)
        newstate._mode = new_mode
        return newstate

    def __iter__(self):
        for elem in self._content:
            yield elem

# }%}

# Moves {%{

class AbstractMove(object):
    def applyMoves(self, state):
        raise NotImplementedError

    def canApplyTo(self, state):
        return False

    def key(self):
        raise NotImplementedError

class MoveInInsertMode(AbstractMove):
    def canApplyTo(self, state):
        return state.mode() == EditorState.MODE_INSERT

class MoveInNormalMode(AbstractMove):
    def canApplyTo(self, state):
        return state.mode() == EditorState.MODE_NORMAL

# NormalToInsert {%{

class MoveFromNormalToInsert(MoveInNormalMode):
    pass

class KeyALowerCase(MoveFromNormalToInsert):

    def applyMoves(self, state):
        assert state.mode() == EditorState.MODE_NORMAL
        newstate = state.setMode(EditorState.MODE_INSERT)
        ret = [(1, newstate)]

        if newstate.position() != state.lastPos():
            if not newstate.chars(newstate.position()).mustContain({'\n'}):
                newstate = newstate.forceNot(newstate.position(), {'\n'})
                newstate = newstate.setPosition(newstate.position()+1)
                ret.append((1, newstate))

        return ret

    def key(self):
        return 'a'

class KeyAUpperCase(MoveFromNormalToInsert):

    def applyMoves(self, state):
        assert state.mode() == EditorState.MODE_NORMAL

        newstate = state.setMode(EditorState.MODE_INSERT)

        pos = newstate.position()
        ret = []

        # end of line or not? we can end up everywhere
        while pos < state.lastPos() and not state.chars(pos).mustContain({'\n'}):

            if state.chars(pos).canContain({'\n'}):
                # we have to split the states
                otherstate = newstate.force(pos, {'\n'})
                otherstate = otherstate.setPosition(pos)
                ret.append((1,otherstate))

            newstate = newstate.forceNot(pos, {'\n'})

            pos += 1

        pos += 1

        if pos > state.lastPos():
            pos = state.lastPos()

        newstate = newstate.setPosition(pos)
        ret.append((1,newstate))

        return ret

    def key(self):
        return 'A'
# }%}

# Normal {%{

class NormalModeMove(MoveInNormalMode):
    pass

class KeyXLowerCase(NormalModeMove):

    def applyMoves(self, state):
        assert state.mode() == EditorState.MODE_NORMAL
        position = state.position()

        # empty text => no possible deletion
        if state.lastPos() == 0 or state.position() >= state.lastPos():
            return []

        ret = []

        # option (1), the character after the current car is not an EOL
        newstate = state.remove(position)

        if position < newstate.lastPos():
            if not newstate.chars(position).mustContain({'\n'}):
                ret.append((1,newstate.forceNot(position, {'\n'})))
        elif position == 0 and newstate.lastPos() == 0:
            ret.append((1,newstate))

        # option (2) it's an end of line
        if position > 0 and not newstate.chars(position-1).mustContain({'\n'}):

            newstate = newstate.forceNot(position-1, {'\n'})

            if position < newstate.lastPos() and not newstate.chars(position).canContain({'\n'}):
                return ret

            if position < newstate.lastPos():
                newstate = newstate.force(position, {'\n'})

            newstate = newstate.setPosition(position-1)
            ret.append((1,newstate))

        return ret

    def key(self):
        return 'x'

class MoveToTheBeginningOfWord(object):

    def __init__(self, increment):
        self._increment = increment

    def increment(self):
        return self._increment

    def applyMoves(self, state):
        assert state.mode() == EditorState.MODE_NORMAL
        ret = []

        if state.lastPos() == 0:
            assert state.position() == 0
            return [state]

        if state.lastPos() == state.position():
            e_pos = state.position()
            assert state.chars(e_pos-1).mustContain(SPACE_CHARS)
            if self.increment() > 0:
                return [state]
            state = state.setPosition(e_pos-1)

        if state.position() == 0 and self.increment() < 0:
            ret.append(state)

        pos = state.position()
        pos += self.increment()

        while 0 <= pos <= state.lastPos():

            if pos == 0 and self.increment() < 0:
                ret.append(state.setPosition(pos))
                break
            if pos == state.lastPos() and self.increment() > 0:
                if state.chars(pos-1).mustContain({'\n'}):
                    ret.append(state.setPosition(pos))
                else:
                    ret.append(state.forceNot(pos-1, {'\n'}).setPosition(pos-1))
                    if state.chars(pos-1).canContain({'\n'}):
                        ret.append(state.force(pos-1, {'\n'}).setPosition(pos))
                break

            can_space = pos == state.lastPos() or state.chars(pos).canContain(SPACE_CHARS)
            must_space = pos == state.lastPos() or state.chars(pos).mustContain(SPACE_CHARS)

            if not can_space:
                ret.append(state.setPosition(pos))
                break
            elif not must_space:
                state_word = state.forceNot(pos, SPACE_CHARS)
                ret.append(state_word.setPosition(pos))
                state = state.force(pos, SPACE_CHARS)

            pos = pos + self.increment()

        assert len(ret) > 0
        return ret

class MoveToEndOfWord(object):

    def __init__(self, increment):
        self._increment = increment

    def increment(self):
        return self._increment

    def applyMoves(self, state):
        assert state.mode() == EditorState.MODE_NORMAL

        if state.position() == 0 and self.increment() < 0 or \
            state.position() >= state.lastPos() - 1 and self.increment() > 0:
            return [state]

        assert not state.chars(state.position()).canContain(SPACE_CHARS)

        pos = state.position()
        pos += self.increment()
        out = False
        ret = []

        while not out:
            out = not (0 <= pos < state.lastPos())

            if out:
                break

            can_space = state.chars(pos).canContain(SPACE_CHARS)
            must_space = state.chars(pos).mustContain(SPACE_CHARS)

            if must_space or can_space:
                pos_before = pos - self.increment()
                state_end = state.setPosition(pos_before)
                if not out and not must_space:
                    state_end = state_end.force(pos_before+self.increment(), SPACE_CHARS)
                ret.append(state_end)

                if must_space:
                    break

                state = state.forceNot(pos, SPACE_CHARS)

            pos += self.increment()

        if out:
            pos -= self.increment()
            ret.append(state.setPosition(pos))

        return ret

class MoveToSpecificChar(object):

    def __init__(self, increment, elements_before):
        self._increment = increment
        self._elements_before = elements_before

    def increment(self):
        return self._increment

    def applyMoves(self, state):
        assert state.mode() == EditorState.MODE_NORMAL
        e_pos = state.position()

        if e_pos == state.lastPos():
            assert e_pos == 0 or state.chars(e_pos-1).mustContain(self._elements_before)
            return []

        while e_pos + self.increment() < state.lastPos() and e_pos + self.increment() > 0:
            if state.chars(e_pos).mustContain(self._elements_before):
                break

            e_pos += self.increment()

        # we are at the right position
        e_pos += -self.increment()

        ret = []

        newstate = EditorState.copy(state)

        for endpos in range(state.position()+self.increment(),
                            e_pos+self.increment(),
                            self.increment()):

            newstate = newstate.forceNot(endpos, self._elements_before)

            thisstate = EditorState.copy(newstate)

            key = thisstate.chars(endpos)

            #TODO: Add a constraint
            # no key in the previous values
            # for intermediate_key in range(state.position()+1, endpos):
                # thisstate = thisstate.forceNot(intermediate_key, key)

            thisstate = thisstate.setPosition(endpos)
            ret.append(thisstate)

        return ret

class MoveBefore(object):

    def __init__(self, moveToChar):
        self._moveToChar = moveToChar

    def applyMoves(self, state):
        currentPosition = state.position()
        ret = []

        for newstate in self._moveToChar.applyMoves(state):
            keep = False

            if self._moveToChar.increment() > 0:
                keep = newstate.position() > currentPosition + 1
            else:
                keep = newstate.position() < currentPosition - 1

            if keep:
                ret.append(newstate.setPosition(newstate.position()-1))

        return ret

class KeyQuickMove(NormalModeMove):

    def __init__(self, nbkeys, moveToChar, name):
        self._nbkeys = nbkeys
        if not isinstance(moveToChar, list):
            moveToChar = [moveToChar]
        self._moveToChar = moveToChar
        self._name = name

    def applyMoves(self, state):
        states = [state]

        for move in self._moveToChar:
            newstates = []
            for state in states:
                newstates.extend(move.applyMoves(state))

            states = newstates

        return map(lambda x : (self._nbkeys, x), states)

    def key(self):
        return self._name


class KeyQuickDelete(NormalModeMove):

    def __init__(self, nbkeys, moveToChar, name):
        self._nbkeys = nbkeys
        if not isinstance(moveToChar, list):
            moveToChar = [moveToChar]
        self._moveToChar = moveToChar
        self._name = name

    def applyMoves(self, state):
        baseState = state
        states = [state]

        for move in self._moveToChar:
            newstates = []
            for state in states:
                newstates.extend(move.applyMoves(state))

            states = newstates

        base = state.position()
        newstates = []

        for state in states:
            # no change
            assert state.lastPos() == baseState.lastPos()

            if state.position() != baseState.position():
                fromPos = min(state.position(), baseState.position())
                toPos = max(state.position(), baseState.position())
                pass

        return []

    def key(self):
        return self._name

# }%}

# Insert {%{
class InsertMode(MoveInInsertMode):
    pass
# }%}

# InsertToNormal {%{
class MoveFromInsertToNormal(MoveInInsertMode):
    pass

class TextEsc(MoveFromInsertToNormal):

    def key(self):
        return '<ESC>'

    def applyMoves(self, state):
        #TODO: It seems that we move backward when we press "ESC"
        assert state.mode() == EditorState.MODE_INSERT

        newPosition = state.position() - 1
        ret = []

        if newPosition < 0 or state.chars(newPosition).mustContain({'\n'}):
            state = state.setPosition(max(0, newPosition))
            state = state.setMode(EditorState.MODE_NORMAL)
            ret.append((1, state))
        else:

            if newPosition >= 0 and state.chars(newPosition).canContain({'\n'}):
                state = state.setMode(EditorState.MODE_NORMAL)
                ret.append((1, state.force(newPosition, {'\n'})))

            state = state.setMode(EditorState.MODE_NORMAL)
            state = state.setPosition(newPosition)
            ret.append((1, state.forceNot(newPosition, {'\n'})))

        return ret

class TextKey(MoveInInsertMode):

    def key(self):
        return '_'

    def applyMoves(self, state):
        #TODO: What about backspace?
        assert state.mode() == EditorState.MODE_INSERT
        stateId = state.nextState()
        return [(1, state.insert(state.position(), AnyCharPossibility(stateId)))]

# }%}

# }%}

def applyMoves(states, moves):
    next_states = []

    for nbmoves, lastmoves, state in states:

        rightMoves = filter(lambda x : x.canApplyTo(state), moves)

        for move in rightMoves:
            new_states = move.applyMoves(state)

            for newmoves, new_state in new_states:
                addmove = lastmoves.copy()
                addmove.append(move)
                next_states.append((nbmoves+newmoves, addmove, new_state))

    return next_states

def compress_states(states):
    import collections

    state_by_length = collections.defaultdict(lambda : [])

    for state in states:
        state_by_length[state[2].lastPos()].append(state)

    states_to_remove = set([])

    for length, states in state_by_length.items():
        for state1 in states:
            for state2 in states:
                is_included = True

                for elem1, elem2 in zip(state1[2], state2[2]):
                    is_included = is_included and elem1 <= elem2
                    if not is_included:
                        break

                if is_included:
                    states_to_remove.add(state1[2])
                    break

    elems = []
    for states in state_by_length.values():
        elems.extend(filter(lambda x : x[2] in states_to_remove, states))
    return elems

def complex_algorithm(str_from, str_to, upper_bound):
    '''
    (1) init state
    (2) apply moves
        (2.1) drop any moves which are above the "easy algorithm"
        (2.2) limit the moves (only letters which are part of the sum)
        (2.3) limit the same states
    '''
    moves = []
    moves.append(KeyALowerCase())
    moves.append(KeyAUpperCase())
    moves.append(KeyXLowerCase())

    for movePossibility in [KeyQuickMove, KeyQuickDelete]: # + Yank, Change
        moves.append(movePossibility(2, MoveToSpecificChar(+1, {'\n'}), 'f?'))
        moves.append(movePossibility(2, MoveToSpecificChar(-1, {'\n'}), 'F?'))
        moves.append(movePossibility(2, MoveBefore(MoveToSpecificChar(+1, {'\n'})), 't?'))
        moves.append(movePossibility(2, MoveBefore(MoveToSpecificChar(-1, {'\n'})), 'T?'))

        moves.append(movePossibility(1, [MoveToTheBeginningOfWord(-1), MoveToEndOfWord(-1)], 'B'))
        moves.append(movePossibility(1, [MoveToTheBeginningOfWord(+1), MoveToEndOfWord(+1)], 'E'))

    moves.append(TextKey())
    moves.append(TextEsc())

    baseState = EditorState(str_from)

    states = [(0, [], baseState)]

    if baseState.canMatch(str_to):
        state = states[0]
        return (state[1], state[0])

    while True:

        states = compress_states(states)
        next_states = applyMoves(states, moves)

        states_dict = {}
        for state in next_states:
            previous_state = states_dict.get(state[2], None)

            if previous_state is None or previous_state[0] > state[0]:
                states_dict[state[2]] = state

        states = states_dict.values()

        for state in states:
            if state[2].canMatch(str_to):

                return (state[1], state[0])

class TestOptimizer(unittest.TestCase):

    def testBase(self):
        baseState = EditorState('abc')
        self.assertTrue(baseState.canMatch('abc'))

    def testApplySingleMove_A_Insert(self):
        self.assertEqual(complex_algorithm('abc', 'abcE', 5)[1], 2)
        self.assertEqual(complex_algorithm('abc', 'abcEKKK', 5)[1], 5)
        self.assertEqual(complex_algorithm('abc', 'aEbc', 5)[1], 2)

    def testRemoveCharacter(self):
        self.assertEqual(complex_algorithm('abc', '', 5)[1], 3)

    def testFindChar1(self):
        self.assertEqual(complex_algorithm('abcdefghijkl', 'abdefghikl', 12)[1], 6)

    def testEmpty(self):
        self.assertEqual(complex_algorithm('a', 'a', 12), ([], 0))

    def testFindChar2(self):
        self.assertEqual(complex_algorithm('abc  deff', 'abc  de', 12)[1], 4)

class TestCharSet(unittest.TestCase):

    def testCharComparison(self):
        self.assertTrue(OneCharPossibility(2, 'a') <= AnyCharPossibility(2))
        self.assertTrue(AnyCharPossibility(2) <= AnyCharPossibility(2))
        self.assertTrue(XCharsPossibilities(2, {'a'}) <= XCharsPossibilities(2, {'a', 'b'}))
        self.assertFalse(XCharsPossibilities(2, {'a', 'b'}) <= XCharsPossibilities(2, {'a', 'c'}))
        self.assertFalse(RemovePossibilitiy({'a'}, AnyCharPossibility(1)) <= RemovePossibilitiy({'a', 'b'}, AnyCharPossibility(1)))

if __name__ == '__main__':
    unittest.main()

