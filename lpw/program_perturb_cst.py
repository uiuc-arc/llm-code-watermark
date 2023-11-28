"""
Code for perturbing Python programs. Taken and modified from https://github.com/jjhenkel/averloc.
"""
import os
import io
import sys
import ast
import libcst as cst
import json
import gzip
import copy
import tqdm
import astor
import random
import itertools
import multiprocessing

# from human_eval.data import write_jsonl, read_problems
from mxeval.data import write_jsonl, read_problems, get_data



def find_comments_in_code(og_code):
   module = cst.parse_module(og_code)

  # Define a visitor to collect comments
   class CommentCollector(cst.CSTVisitor):
      def __init__(self):
          self.comments = 0

      def visit_Comment(self, node):
          self.comments += 1
          

   collector = CommentCollector()
   module.visit(collector)
   return collector.comments != 0

def remove_comments(og_code):

  class CommentRemover(cst.CSTTransformer):
    def leave_Comment(self, original_node, updated_node):
        return cst.RemoveFromParent()
    
  module = cst.parse_module(og_code)
  module.visit(CommentRemover())
  return module.code



def t_rename_variable_in_iterator(module, uid = 1):
  class VariableVisitor(cst.CSTVisitor):
    def __init__(self):
       self.iter_vars = []
    def  visit_For(self, node):
          try:
            self.iter_vars.append(node.target.value)
          except:
             pass
    def visit_While(self, node):
       if hasattr(node, "target"):
          self.iter_vars.append(node.target.value)
    

  
  class VariableRenamer(cst.CSTTransformer):
    def __init__(self, selection):
      self.selection = selection
      self.uid = uid
      
    def leave_For(self, original_node, updated_node):
      return self._rename_loop_variables(updated_node)

    def leave_While(self, original_node, updated_node):
      return self._rename_loop_variables(updated_node)
    



    def _rename_loop_variables(self, node):
      updated_node = node
      if hasattr(node, "target") and isinstance(node.target, cst.Name) and node.target.value == self.selection:
          updated_node = node.with_changes(target=cst.Name(f"REPLACEME{self.uid}"))

      return updated_node.visit(VariableReferenceRenamer(self.selection))
    




  class VariableReferenceRenamer(cst.CSTTransformer):
    def __init__(self, selection):
        self.selection = selection
        self.uid = uid
        

    def leave_Name(self, original_node, updated_node):
        if updated_node.value == self.selection:
            return updated_node.with_changes(value= f"REPLACEME{self.uid}")
        return updated_node
  
  visitor = VariableVisitor()
  module.visit(visitor)
  iter_vars = visitor.iter_vars
  if len(iter_vars) == 0:
     return False, module.code
  selection = random.choice(iter_vars)
  transformer = VariableRenamer(selection)
  module = module.visit(transformer)
  return True, module.code


  

    

def t_rename_parameters(module, uid=1):
  class FunctionParameterCollector(cst.CSTVisitor):
    def __init__(self):
        self.function_parameters = {}
        self.idx = 0

    def visit_FunctionDef(self, node):
        function_name = node.name.value
        parameters = [param.name.value for param in node.params.params if param != 'self']
        self.function_parameters[self.idx] = (function_name, parameters)
        self.idx += 1
    
  class ParameterNameReplacer(cst.CSTTransformer):
      def __init__(self, selection):
          self.uid = uid
          self.selection = selection
          super().__init__()

      
      def leave_Param(self, node: cst.Param, updated_node: cst.Name) -> cst.Param:
        if updated_node.name.value == self.selection:
             return updated_node.with_changes(value= f"REPLACEME{self.uid}")
        return updated_node
      
      
      def leave_Name(self, node: cst.Name, updated_node: cst.Name) -> cst.CSTNode:
        if updated_node.value == self.selection:
          return updated_node.with_changes(value= f"REPLACEME{self.uid}")
        
        return updated_node

  visitor = FunctionParameterCollector()

  module.visit(visitor)

  params = visitor.function_parameters[0][1]

  if len(params) == 0:
    return False, module.code
  
  selection = random.choice(params)

  
  transformer = ParameterNameReplacer(selection)

  module = module.visit(transformer)


  return True, module.code
  




def t_rename_local_variables(module, uid=1):
  
  class FunctionParameterCollector(cst.CSTVisitor):
    def __init__(self):
        self.function_parameters = set()

    def visit_FunctionDef(self, node):
        for param in node.params.params:
           self.function_parameters.add(param)
        
  
  
  class VariableNameVisitor(cst.CSTVisitor):
     def __init__(self, function_parameters):
          self.names = set()
          self.function_parameters = function_parameters
          super().__init__()
     
     def visit_Assign(self, node: cst.Name) -> cst.CSTNode:
          for target in node.targets:
             try:
              if target.target.value not in self.function_parameters:
                self.names.add(target.target.value)
             except:
                continue


  class VariableNameReplacer(cst.CSTTransformer):
      def __init__(self, selection):
          self.uid = uid
          self.selection = selection
          super().__init__()

      def leave_Name(self, node: cst.Name, updated_node: cst.Name) -> cst.CSTNode:

        if updated_node.value == self.selection:
          return updated_node.with_changes(value= f"REPLACEME{self.uid}")
        
        return updated_node
  
  
  param_visitor = FunctionParameterCollector()
  module.visit(param_visitor)

  visitor = VariableNameVisitor(param_visitor.function_parameters)

  module.visit(visitor)

  if len(visitor.names) == 0:
     return False, module.code

  selection = random.choice(list(visitor.names))

  
  transformer = VariableNameReplacer(selection)

  module = module.visit(transformer)


  return True, module.code




def t_unroll_whiles(module, uid=1):
  
  the_ast = ast.parse(module.code)

  if len(the_ast.body) == 0 or not isinstance(the_ast.body[0], ast.FunctionDef):
    return False, astor.to_source(the_ast)
  
 
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
    return False, astor.to_source(the_ast)
 
  return changed, astor.to_source(UnrollWhiles(random.randint(0, count - 1)).visit(the_ast))







def t_wrap_try_catch(module, uid=1):
  if len(module.body) == 0:
    return False, module.code


  class SimpleStatementVisitor(cst.CSTVisitor):
     def __init__(self):
          self.count = 0
          super().__init__()
     
     def visit_SimpleStatementLine(self, node: cst.Name) -> cst.CSTNode:
          self.count += 1
  
  class AddPrintStatements(cst.CSTTransformer):
      def __init__(self, selection):
          self.uid = uid
          self.selection = selection
          self.count = 0
          super().__init__()

      def leave_SimpleStatementLine(self, node: cst.Name, updated_node: cst.Name) -> cst.CSTNode:
          self.count += 1
          if self.count == self.selection:
            

            return cst.Try(
              body= cst.IndentedBlock([updated_node]),
              handlers=[
                  cst.ExceptHandler(
                      body= cst.IndentedBlock([
                            cst.parse_module(f"REPLACEME{self.uid} = 1 \n")
                        ])
                    )
                ]

          )

          return updated_node
      
  visitor = SimpleStatementVisitor()

  module.visit(visitor)

  if visitor.count == 0:
     return False, module.code

  transformer = AddPrintStatements(visitor.count)

  module = module.visit(transformer)

  return True, module.code

def t_add_dead_code(module, uid=1):
  if len(module.body) == 0:
    return False, module.code

  class SimpleStatementVisitor(cst.CSTVisitor):
     def __init__(self):
          self.count = 0
          super().__init__()
     
     def visit_SimpleStatementLine(self, node: cst.Name) -> cst.CSTNode:
          self.count += 1
  
  class AddPrintStatements(cst.CSTTransformer):
      def __init__(self, selection):
          self.uid = uid
          self.selection = selection
          self.count = 0
          super().__init__()

      def leave_SimpleStatementLine(self, node: cst.Name, updated_node: cst.Name) -> cst.CSTNode:
          self.count += 1
          if self.count == self.selection:
            new_code = cst.parse_module(f"""if False: \n  REPLACEME{self.uid} = 1""").body

            return cst.FlattenSentinel([*new_code, updated_node])
          

          return updated_node
      
  
  visitor = SimpleStatementVisitor()

  module.visit(visitor)

  if visitor.count == 0:
     return False, module.code


  transformer = AddPrintStatements(visitor.count)

  module = module.visit(transformer)

  return True, module.code

  


def t_insert_print_statements(module, uid=1):
  if len(module.body) == 0:
    return False, module.code

  
  class SimpleStatementVisitor(cst.CSTVisitor):
     def __init__(self):
          self.count = 0
          super().__init__()
     
     def visit_SimpleStatementLine(self, node: cst.Name) -> cst.CSTNode:
          self.count += 1
  
  class AddPrintStatements(cst.CSTTransformer):
      def __init__(self, selection):
          self.uid = uid
          self.selection = selection
          self.count = 0
          super().__init__()

      def leave_SimpleStatementLine(self, node: cst.Name, updated_node: cst.Name) -> cst.CSTNode:
          self.count += 1
          if self.count == self.selection:
            new_code = cst.parse_module(f"print('REPLACEME{self.uid}')\n").body
            return cst.FlattenSentinel([updated_node, *new_code])
          return updated_node
      
  
  visitor = SimpleStatementVisitor()
  module.visit(visitor)

  if visitor.count == 0:
     return False, module.code

  
  transformer = AddPrintStatements(random.randint(1, visitor.count))
  module = module.visit(transformer)
  return True, module.code


def t_replace_true_false(module, uid=1):
  class BoolVisitCounter(cst.CSTVisitor):
      def __init__(self):
          self.count = 0
          super().__init__()

      def visit_Name(self, node: cst.Name) -> cst.CSTNode:
           if node.value in {'True', 'False'}:
              self.count += 1
           

  class ReplaceTrueFalse(cst.CSTTransformer):
      def __init__(self, selection):
          self.uid = uid
          self.selection = selection
          self.count = 0
          super().__init__()

      def leave_Name(self, node: cst.Name, updated_node: cst.Name) -> cst.CSTNode:
          if node.value in {'True', 'False'}:
            self.count += 1
          
          if self.count == self.selection:
            if updated_node.value == 'True':
                expr = f"'REPLACEME{self.uid}' == 'REPLACEME{self.uid}'"
                self.count += 1
                return cst.parse_expression(expr)

                
            elif updated_node.value == 'False':
              expr = f"'REPLACEME{self.uid}' != 'REPLACEME{self.uid}'"
              self.count += 1
              return cst.parse_expression(expr)

          return updated_node

  visitor = BoolVisitCounter()

  module.visit(visitor)

  if visitor.count == 0:
     return False, module.code

  selection = random.randint(1, visitor.count)
  transformer = ReplaceTrueFalse(selection)

  module = module.visit(transformer)

  return True, module.code


class t_seq(object):
  def __init__(self, transforms):
    self.transforms = transforms
  def __call__(self, the_ast):
    did_change = False
    cur_ast = the_ast
    for i,t in enumerate(self.transforms):
      
      cur_ast = cst.parse_module(cur_ast)
      changed, cur_ast = t(cur_ast, i+1) 
      
      if changed:
        did_change = True
    return did_change, cur_ast





def t_identity(the_ast):
  return True, the_ast


def perturb(og_code, int_id = None, depth = 1, samples = 1):
    transforms = []
    DEPTH = depth
    NUM_SAMPLES = samples

    for s in range(NUM_SAMPLES):
      the_seq = []
      for i in range(DEPTH):
        if not int_id:
          int_id = random.randint(1, 8)
        if int_id == 0:
          the_seq.append(t_identity)
        if int_id == 1:
          if 'True' not in og_code and 'False' not in og_code:
             the_seq.append(t_add_dead_code)
          the_seq.append(t_replace_true_false)
        elif int_id == 2:
          if i % 3 == 0:
            the_seq.append(t_rename_parameters)
          elif i % 3 == 1:
            the_seq.append(t_rename_variable_in_iterator)
          else:
            the_seq.append(t_rename_local_variables)
        elif int_id == 3:
          the_seq.append(t_rename_parameters)
        elif int_id == 4:
          the_seq.append(t_rename_parameters)
        elif int_id == 5:
          the_seq.append(t_insert_print_statements)
        elif int_id == 6:
          the_seq.append(t_add_dead_code)
        elif int_id == 7:
          if 'while' not in og_code:
             the_seq.append(t_add_dead_code)
          the_seq.append(t_unroll_whiles)
        elif int_id == 8:
          the_seq.append(t_wrap_try_catch)

      transforms.append(('depth-{}-sample-{}'.format(DEPTH, s+1), t_seq(the_seq), the_seq))
      
    results = []
    for t_name, t_func, the_seq in transforms:
        try:
            exec(og_code)
        except Exception as ex:
            import traceback
            traceback.print_exc()
            results.append({'changed': False, 't_name': t_name, 'the_seq': the_seq, 'result': og_code})
            continue
        
        try:
           cst.parse_module(og_code)
        except:
           og_code = astor.to_source(ast.parse(og_code))

        changed, result = t_func(og_code)

        results.append({'changed': changed, 't_name': t_name, 'the_seq': the_seq, 'result': result})
    return results


if __name__ == "__main__":
    problems = get_data()
    n = 150
    keys = list(problems.keys())[:n]

    for i in range(n):
        og_code = problems[keys[i]]['prompt'] + problems[keys[i]]['canonical_solution']
        res = perturb(og_code)
        print('original code:')
        print(og_code)
        print('perturbed code 1:' + str(res[0]['the_seq']))
        print(res[0]['result'])

        




        