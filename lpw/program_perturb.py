"""
Code for perturbing Python programs. Taken and modified from https://github.com/jjhenkel/averloc.
"""
import os
import io
import sys
import ast
import json
import gzip
import copy
import tqdm
import astor
import random
import itertools
import multiprocessing

from human_eval.data import write_jsonl, read_problems

PY_2_BUILTINS = [
  'bytearray',
  'IndexError',
  'all',
  'help',
  'vars',
  'SyntaxError',
  'unicode',
  'UnicodeDecodeError',
  'memoryview',
  'isinstance',
  'copyright',
  'NameError',
  'BytesWarning',
  'dict',
  'input',
  'oct',
  'bin',
  'SystemExit',
  'StandardError',
  'format',
  'repr',
  'sorted',
  'False',
  'RuntimeWarning',
  'list',
  'iter',
  'reload',
  'Warning',
  '__package__',
  'round',
  'dir',
  'cmp',
  'set',
  'bytes',
  'reduce',
  'intern',
  'issubclass',
  'Ellipsis',
  'EOFError',
  'locals',
  'BufferError',
  'slice',
  'FloatingPointError',
  'sum',
  'getattr',
  'abs',
  'exit',
  'print',
  'True',
  'FutureWarning',
  'ImportWarning',
  'None',
  'hash',
  'ReferenceError',
  'len',
  'credits',
  'frozenset',
  '__name__',
  'ord',
  'super',
  '_',
  'TypeError',
  'license',
  'KeyboardInterrupt',
  'UserWarning',
  'filter',
  'range',
  'staticmethod',
  'SystemError',
  'BaseException',
  'pow',
  'RuntimeError',
  'float',
  'MemoryError',
  'StopIteration',
  'globals',
  'divmod',
  'enumerate',
  'apply',
  'LookupError',
  'open',
  'quit',
  'basestring',
  'UnicodeError',
  'zip',
  'hex',
  'long',
  'next',
  'ImportError',
  'chr',
  'xrange',
  'type',
  '__doc__',
  'Exception',
  'tuple',
  'UnicodeTranslateError',
  'reversed',
  'UnicodeEncodeError',
  'IOError',
  'hasattr',
  'delattr',
  'setattr',
  'raw_input',
  'SyntaxWarning',
  'compile',
  'ArithmeticError',
  'str',
  'property',
  'GeneratorExit',
  'int',
  '__import__',
  'KeyError',
  'coerce',
  'PendingDeprecationWarning',
  'file',
  'EnvironmentError',
  'unichr',
  'id',
  'OSError',
  'DeprecationWarning',
  'min',
  'UnicodeWarning',
  'execfile',
  'any',
  'complex',
  'bool',
  'ValueError',
  'NotImplemented',
  'map',
  'buffer',
  'max',
  'object',
  'TabError',
  'callable',
  'ZeroDivisionError',
  'eval',
  '__debug__',
  'IndentationError',
  'AssertionError',
  'classmethod',
  'UnboundLocalError',
  'NotImplementedError',
  'AttributeError',
  'OverflowError'
]


def t_rename_fields(the_ast, uid=1):
  changed = False

  # Going to need parent info
  for node in ast.walk(the_ast):
    for child in ast.iter_child_nodes(node):
        child.parent = node

  candidates = []
  for node in ast.walk(the_ast):
    if isinstance(node, ast.Name) and node.id == 'self':
      if isinstance(node.parent, ast.Attribute):
        if isinstance(node.parent.parent, ast.Call) and node.parent.parent.func == node.parent:
          continue
        if node.parent.attr not in [ c.attr for c in candidates ]:
          candidates.append(node.parent)

  if len(candidates) == 0:
    return False, the_ast

  selected = random.choice(candidates)

  to_rename = []
  for node in ast.walk(the_ast):
    if isinstance(node, ast.Name) and node.id == 'self':
      if isinstance(node.parent, ast.Attribute) and node.parent.attr == selected.attr:
        if isinstance(node.parent.parent, ast.Call) and node.parent.parent.func == node.parent:
          continue
        to_rename.append(node.parent)

  for node in to_rename:
    changed = True
    node.attr = 'REPLACEME' + str(uid)

  return changed, the_ast


def t_rename_parameters(the_ast, uid=1):
  changed = False
  candidates = []
  for node in ast.walk(the_ast):
    if isinstance(node, ast.arg):
      if node.arg != 'self' and node.arg not in [ c.arg for c in candidates ]:
        # print(node.arg, node.lineno)
        candidates.append(node)

  if len(candidates) == 0:
    return False, the_ast

  selected = random.choice(candidates)
  parameter_defs = {}
  parameter_defs[selected.arg] = selected

  to_rename = []
  for node in ast.walk(the_ast):
    if isinstance(node, ast.Name) and node.id in parameter_defs:
      to_rename.append(node)
    elif isinstance(node, ast.arg) and node.arg in parameter_defs:
      to_rename.append(node)
  
  for node in to_rename:
    changed = True
    if hasattr(node, 'arg'):
      node.arg = 'REPLACEME' + str(uid)
    else:
      node.id = 'REPLACEME' + str(uid)

  return changed, the_ast


def t_rename_local_variables(the_ast, uid=1):
  changed = False
  candidates = []
  for node in ast.walk(the_ast):
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
      if node.id not in [ c.id for c in candidates ]:
        # print(node.id, node.lineno)
        candidates.append(node)

  if len(candidates) == 0:
    return False, the_ast


  selected = random.choice(candidates)
  local_var_defs = {}
  local_var_defs[selected.id] = selected

  to_rename = []
  for node in ast.walk(the_ast):
    if isinstance(node, ast.Name) and node.id in local_var_defs:
      to_rename.append(node)
  
  for node in to_rename:
    changed = True
    node.id = 'REPLACEME' + str(uid)

  return changed, the_ast


def t_unroll_whiles(the_ast, uid=1):
  if len(the_ast.body) == 0 or not isinstance(the_ast.body[0], ast.FunctionDef):
    return False, the_ast
  
  class UnrollWhiles(ast.NodeTransformer):
    def __init__(self, selection):
      self.selection = selection
      self.count = 0
      self.done = False
      super().__init__()

    def visit_While(self, node):
      if self.done:
        return node
      if self.count != self.selection:
        self.count += 1
        return node
      
      self.done = True
      return ast.While(
        test=node.test,
        body=node.body + [ node, ast.Break() ],
        orelse=[]
      )

  changed = False
  count = 0
  
  for node in ast.walk(the_ast):
    if isinstance(node, ast.While):
      changed = True
      count += 1

  if count == 0:
    return False, the_ast
 
  return changed, UnrollWhiles(random.randint(0, count - 1)).visit(the_ast)


def t_wrap_try_catch(the_ast, uid=1):
  if len(the_ast.body) == 0 or not isinstance(the_ast.body[0], ast.FunctionDef):
    return False, the_ast

  temp = ast.Try(
    body = the_ast.body[0].body,
    handlers=[ast.ExceptHandler(
      type=ast.Name(id='Exception', ctx=ast.Load()),
      name='REPLACME' + str(uid),
      body=[ast.Raise()]
    )],
    orelse=[],
    finalbody=[]
  )

  the_ast.body[0].body = [temp]

  return True, the_ast


def t_add_dead_code(the_ast, uid=1):
  if len(the_ast.body) == 0 or not isinstance(the_ast.body[0], ast.FunctionDef):
    return False, the_ast

  if bool(random.getrandbits(1)):
    the_ast.body[0].body.insert(
      0,
      ast.If(
        test=ast.Name(id="False", ctx=ast.Load()),
        body=[ 
          ast.Assign(
            targets=[ast.Name(id="REPLACME" + str(uid), ctx=ast.Store())],
            value=ast.Num(n=1)
          )
        ],
        orelse=[]
      )
    )
  else:
    the_ast.body[0].body.append(
      ast.If(
        test=ast.Name(id="False", ctx=ast.Load()),
        body=[ 
          ast.Assign(
            targets=[ast.Name(id="REPLACME" + str(uid), ctx=ast.Store())],
            value=ast.Num(n=1)
          )
        ],
        orelse=[]
      )
    )
  
  return True, the_ast


def t_insert_print_statements(the_ast, uid=1):
  if len(the_ast.body) == 0 or not isinstance(the_ast.body[0], ast.FunctionDef):
    return False, the_ast

  if bool(random.getrandbits(1)):
    the_ast.body[0].body.insert(
      0,
      ast.Expr(
        ast.Call(
          func=ast.Name(id="print", ctx=ast.Load()),
          args=[ast.Str("REPLACEME" + str(uid))],
          keywords=[]
        )
      )
    )
  else:
    the_ast.body[0].body.append(
      ast.Expr(
        ast.Call(
          func=ast.Name(id="print", ctx=ast.Load()),
          args=[ast.Str("REPLACEME" + str(uid))],
          keywords=[]
        )
      )
    )
  
  return True, the_ast


def t_replace_true_false(the_ast, uid=1):
  class ReplaceTrueFalse(ast.NodeTransformer):
    def __init__(self, selection):
      self.selection = selection
      self.count = 0
      self.done = False
      super().__init__()

    def visit_NameConstant(self, node):
      if self.done:
        return node
      if node.value != True and node.value != False:
        return node
      if self.count != self.selection:
        self.count += 1
        return node
      self.done = True
      return ast.Compare(
        left=ast.Str("REPLACEME" + str(uid)),
        ops=[ast.Eq() if node.value == True else ast.NotEq()],
        comparators=[ast.Str("REPLACEME" + str(uid))]
      ) 

  changed = False
  count = 0
  
  for node in ast.walk(the_ast):
    if isinstance(node, ast.NameConstant) and node.value == True:
      changed = True
      count += 1
    elif isinstance(node, ast.NameConstant) and node.value == False:
      changed = True
      count += 1

  if count == 0:
    return False, the_ast
 
  return changed, ReplaceTrueFalse(random.randint(0, count - 1)).visit(the_ast)


class t_seq(object):
  def __init__(self, transforms):
    self.transforms = transforms
  def __call__(self, the_ast):
    did_change = False
    cur_ast = the_ast
    for i,t in enumerate(self.transforms):
      changed, cur_ast = t(cur_ast, i+1) 
      if changed:
        did_change = True
    return did_change, cur_ast


def t_identity(the_ast):
  return True, the_ast


def perturb(og_code, int_id = None, depth = 1, samples = 1):
    transforms = [
        (
        'transforms.Identity',
        t_identity,
        'Identity'
        )
    ]

    DEPTH = depth
    NUM_SAMPLES = samples

    for s in range(NUM_SAMPLES):
      the_seq = []
      for _ in range(DEPTH):
        if not int_id:
          int_id = random.randint(1, 8)
        if int_id == 1:
          the_seq.append(t_replace_true_false)
        elif int_id == 2:
          the_seq.append(t_rename_local_variables)
        elif int_id == 3:
          the_seq.append(t_rename_parameters)
        elif int_id == 4:
          the_seq.append(t_rename_fields)
        elif int_id == 5:
          the_seq.append(t_insert_print_statements)
        elif int_id == 6:
          the_seq.append(t_add_dead_code)
        elif int_id == 7:
          the_seq.append(t_unroll_whiles)
        elif int_id == 8:
          the_seq.append(t_wrap_try_catch)
    
      transforms.append(('depth-{}-sample-{}'.format(DEPTH, s+1), t_seq(the_seq), the_seq))
      
    results = []
    for t_name, t_func, the_seq in transforms:
        try:
            # t_func = t_seq(the_seq)
            changed, result = t_func(
                ast.parse(og_code)
            )
            results.append({'changed': changed, 't_name': t_name, 'the_seq': the_seq, 'result': astor.to_source(result)})
        except Exception as ex:
            import traceback
            traceback.print_exc()
        results.append({'changed': False, 't_name': t_name, 'the_seq': the_seq, 'result': og_code})
    return results


if __name__ == "__main__":
    # Get the human-eval data
    problems = read_problems()
    
    # Get the first key
    n = 5
    keys = list(problems.keys())[:n]

    for i in range(n):
        og_code = problems[keys[i]]['prompt'] + problems[keys[i]]['canonical_solution']
        res = perturb(og_code)
        print('original code:')
        print(og_code)
        print('perturbed code 1:' + str(res[0]['the_seq']))
        print(res[0]['result'])
        print('perturbed code 2:' + str(res[1]['the_seq']))
        print(res[1]['result'])
        print('perturbed code 3:' + str(res[2]['the_seq']))
        print(res[2]['result'])
