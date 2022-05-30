# class debug_functions():

#     global verbose 
#     verbose = False

#     def set_verbose(self):
#         global verbose
#         verbose = True

#     def log(self, txt, obj = None):
#         global verbose
#         if verbose:
#             if obj is None:
#                 print('[LOG] ' , txt)
#             else:
#                 print('[LOG] ' , txt, ' : ', obj)
        
#         pass

#     def err(self, txt, error = None):
#         print('[ERROR] ', txt)
#         if error is not None:
#             print(error)

#     def out(self, txt, user = None):
#         if user is None:
#             print('[OUTPUT] ', txt)
#         else:
#             print(user, ':', txt)

#     pass

verbose = False

def set_verbose():
    verbose = True

def log(txt, obj = None):
    if verbose:
        if obj is None:
            print('[LOG] ' , txt)
        else:
            print('[LOG] ' , txt, ' : ', obj)
    
    pass

def err(txt, error = None):
    print('[ERROR] ', txt)
    if error is not None:
        print(error)

def out(txt, user = None):
    if user is None:
        print('[OUTPUT] ', txt)
    else:
        print(user, ':', txt)