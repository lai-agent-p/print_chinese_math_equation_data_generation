class nodes(object):

    def leaf_node(self):
        return {'type': 'leaf',
                'offset': None,
                'actual_offset': None,
                'scale': None,
                'content': None,
                'h': None,
                'w': None,
                'label': None,
                'path': None,
                }

    def frac_node(self):
        return {'type': 'frac',
                'global_offset': [0, 0],
                'num_offset': None,
                'den_offset': None,
                'num': [],
                'den': [],
                'line_sym': {'sym': None,
                             'h': None,
                             'w': None,
                             'lu_pos': None},
                }

    def sqrt_node(self):
        return {'type': 'sqrt',
                'offset': None,
                'sqrt_sym': {'sym': None,
                             'offset': None,
                             'h': None,
                             'w': None,
                             'lu_pos': None},
                'root_num': {'num': None,
                             'offset': None,
                             'h': None,
                             'w': None,
                             'label': None},
                'content': [],
                'line_sym': {'sym': None,
                             'h': None,
                             'w': None,
                             'lu_pos': None},
                }

    def one_arg_node(self):
        return {'type': 'one_arg',
                'left_offset': None,
                'right_offset': None,
                'special_sym': {'sym': None,
                                'offset': None,
                                'h': None,
                                'w': None,
                                'label': None,
                                'lu_pos': None},
                'content': [],
                }

    def scription_node(self):
        return {'type': 'scription',
                'offset': None,
                'content': [],
                'label': None,
                }

    def deletion_node(self):
        return {'type': 'deletion',
                'offset': None,
                'scale': None,
                'content': None,
                'h': None,
                'w': None,
                'label': None,
                }

    def sqrt_deletion_node(self):
        return {'type': 'sqrt_deletion',
                'offset': None,
                'scale': None,
                'content': None,
                'h': None,
                'w': None,
                'label': None,
                }

    def frac_deletion_node(self):
        return {'type': 'frac_deletion',
                'offset': None,
                'scale': None,
                'content': None,
                'h': None,
                'w': None,
                'label': None,
                }

    def move_node(self):
        return {'type': 'move',
                'offset': None,
                'scale': None,
                'content': [],
                'h': None,
                'w': None,
                'label': None,
                'move_loc': None,
                }

    def sqrt_move_node(self):
        return {'type': 'sqrt_move',
                'offset': None,
                'scale': None,
                'content': [],
                'h': None,
                'w': None,
                'label': None,
                'move_loc': None,
                }
