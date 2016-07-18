from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import pyxe.command_parsing as cp


def test_clean():

    examples = {'peak': 'peaks',
                'peak_err': 'peaks error',
                'strain': 'strain',
                'stress': 'stress',
                'strain error': 'strain error',
                'shears_strians': 'shear strain',
                'stresses-err': 'stress error',
                'shears-stresses': 'shear stress'}

    for example in examples:
        cleaned = cp.text_cleaning(example)
        answer = examples[example]
        assert cleaned == answer, 'Clean: {}, Ans: {}'.format(cleaned, answer)


def test_valid_entry():
    examples = {'peaks': True,
                'peaks error': True,
                'fwhm error': True,
                'strain error': True,
                'shear stress error': False,
                'sdafggfd': False,
                ' shear stress': False}

    for example in examples:
        try:
            cp.validate_entry(example)
            valid = True
        except AssertionError:
            valid = False
        answer = examples[example]
        assert valid == answer, 'Check: {}, Ans: {}'.format(valid, answer)


def test_az_command():
    examples = {'peaks': {'phi': False, 'az_idx': True},
                'peaks error': {'phi': False, 'az_idx': True},
                'fwhm': {'phi': False, 'az_idx': True},
                'fwhm error': {'phi': False, 'az_idx': True},
                'strain': {'phi': True, 'az_idx': True},
                'strain error': {'phi': False, 'az_idx': True},
                'shear strain': {'phi': True, 'az_idx': False},
                'stress': {'phi': True, 'az_idx': True},
                'stress error': {'phi': False, 'az_idx': True},
                'shear stress': {'phi': True, 'az_idx': False}}

    for example in examples:
        a = [1, None, None, 1]
        b = [None, 1, None, 1]
        c = [examples[example]['phi'], examples[example]['az_idx'], False, False]

        for phi, az_idx, answer in zip(a, b, c):
            print(phi, az_idx, answer)
            try:
                cp.validate_azimuthal_selection(example, phi, az_idx)
                valid = True
            except AssertionError:
                valid = False
            error = 'Test: {} - {}, Ans: {}'.format(example, valid, answer)
            assert valid == answer, error

def test_validate_command():
    examples = {'peak': {'phi': False, 'az_idx': True},
                'peak_err': {'phi': False, 'az_idx': True},
                'fwhm': {'phi': False, 'az_idx': True},
                'fwhm-error': {'phi': False, 'az_idx': True},
                'strians': {'phi': True, 'az_idx': True},
                'strian_error': {'phi': False, 'az_idx': True},
                'shears strain': {'phi': True, 'az_idx': False},
                'stresses': {'phi': True, 'az_idx': True},
                'stresses error': {'phi': False, 'az_idx': True},
                'shear_stress': {'phi': True, 'az_idx': False},
                'random': {'phi': False, 'az_idx': False}, # inv command
                '_shear stress': {'phi': False, 'az_idx': False}} # inv command

    for example in examples:
        a = [1, None, None, 1]
        b = [None, 1, None, 1]
        c = [examples[example]['phi'], examples[example]['az_idx'], False, False]

        for phi, az_idx, answer in zip(a, b, c):
            print(phi, az_idx, answer)
            try:
                cp.validate_command(example, phi, az_idx)
                valid = True
            except AssertionError:
                valid=False
            error = 'Test: {} - {}, Ans: {}'.format(example, valid, answer)
            assert valid == answer, error


def test_convert_to_analysis():

    # It will return the valid answers to invalid data/command options
    examples = {'peaks': {'phi': 'peaks', 'az_idx': 'peaks'},
                'peaks error': {'phi': 'peaks', 'az_idx': 'peaks'},
                'strain': {'phi': 'strain fit', 'az_idx': 'strain'},
                'strain error': {'phi': 'strain fit', 'az_idx': 'strain'},
                'shear strain': {'phi': 'strain fit', 'az_idx': 'strain'},
                'shear stress': {'phi': 'stress fit', 'az_idx': 'stress'},
                'stress': {'phi': 'stress fit', 'az_idx': 'stress'}}

    for example in examples:
        for az_com in ['phi', 'az_idx']:
            an_lvl = cp.convert_request_to_level(example, az_com)
            answer = examples[example][az_com]
            error = '{} ({}) - {} ({})'.format(example, az_com, an_lvl, answer)
            assert an_lvl == answer, error


def test_analysis_stage():
    examples = [['peaks', 'strain fit', False],
                ['strain', 'strain fit', False],
                ['stress', 'strain fit', False],
                ['stress', 'stress fit', False],
                ['peaks', 'peaks', True],
                ['strain fit', 'strain fit', True],
                ['strain fit', 'strain', True],
                ['stress fit', 'strain', True]]

    for example in examples:
        current, required = example[0], example[1]
        valid = cp.analysis_state_comparison(current, required)
        answer = example[2]
        error = '{} v {} - {} ({})'.format(current, required, valid, answer)
        assert valid == answer, error

if __name__ == '__main__':
    test_clean()
    test_valid_entry()
    test_az_command()
    test_validate_command()
    test_convert_to_analysis()
    test_analysis_stage()