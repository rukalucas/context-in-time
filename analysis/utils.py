"""Analysis utilities for data generation, metadata extraction, and psychometrics."""

import numpy as np
import torch
from pathlib import Path


def compute_trial_events(batch, task):
    """
    Compute event timestamps for all trials from batch metadata.

    Args:
        batch: List of trial dicts (new batch format)
        task: Task instance with dt attribute

    Returns:
        dict mapping event_name -> [num_trials] array of timestep indices
        Available events:
            - 'trial_start': Always 0
            - 'cue_onset': After initial fixation (rule report period)
            - 'rule_report_end': End of rule report response
            - 'timing_start': Start of timing period
            - 'first_pulse': Start of first timing pulse
            - 'decision_start': Start of decision response (= start of second pulse)
            - 'trial_end': End of trial proper (excludes ITI, = iti_start for sequence tasks)
            - 'iti_end': End of ITI (sequence tasks only)
    """
    # Determine number of trials and task type
    if len(batch) > 1:
        # Sequence task: N trial dicts with B=1
        num_trials = len(batch)
        is_sequence = True
    else:
        # Non-sequence task: 1 trial dict with B=num_trials
        num_trials = len(batch[0]['trial_lengths'])
        is_sequence = False

    # Initialize event arrays
    events = {
        'trial_start': np.zeros(num_trials, dtype=int),
        'cue_onset': np.zeros(num_trials, dtype=int),
        'rule_report_end': np.zeros(num_trials, dtype=int),
        'timing_start': np.zeros(num_trials, dtype=int),
        'first_pulse': np.zeros(num_trials, dtype=int),
        'decision_start': np.zeros(num_trials, dtype=int),
        'trial_end': np.zeros(num_trials, dtype=int),
    }

    if is_sequence:
        events['iti_end'] = np.zeros(num_trials, dtype=int)

    # Get task parameters
    rule_report_duration = int(task.rule_report_period / task.dt)
    response_duration = int(task.response_period / task.dt)

    # Extract events for each trial
    for i in range(num_trials):
        # Get trial-specific metadata
        if is_sequence:
            total_length = batch[i]['trial_lengths'][0].item()  # Includes ITI
            initial_fix_rule = batch[i]['metadata']['initial_fixation_rule'][0]
            initial_fix_timing = batch[i]['metadata']['initial_fixation_timing'][0]
            t_m = batch[i]['metadata']['t_m'][0]
            if 'iti_start' in batch[i]['metadata']:
                iti_start = batch[i]['metadata']['iti_start'][0]
                iti_length = batch[i]['metadata']['iti_length'][0]
                # For sequence tasks, actual trial ends at iti_start (not total_length which includes ITI)
                trial_length = int(iti_start)
            else:
                trial_length = total_length
        else:
            trial_length = batch[0]['trial_lengths'][i].item()
            initial_fix_rule = batch[0]['metadata']['initial_fixation_rule'][i]
            initial_fix_timing = batch[0]['metadata']['initial_fixation_timing'][i]
            t_m = batch[0]['metadata']['t_m'][i]

        # Convert to steps
        t_initial_fix_rule = int(initial_fix_rule / task.dt)
        t_initial_fix_timing = int(initial_fix_timing / task.dt)
        t_m_steps = int(t_m / task.dt)

        # Compute event timesteps (all relative to trial start, not including ITI)
        events['trial_start'][i] = 0
        events['cue_onset'][i] = t_initial_fix_rule
        events['rule_report_end'][i] = t_initial_fix_rule + rule_report_duration
        events['timing_start'][i] = events['rule_report_end'][i]
        events['first_pulse'][i] = events['timing_start'][i] + t_initial_fix_timing
        events['decision_start'][i] = trial_length - response_duration
        # Verify timing: decision_start should be first_pulse + t_m
        assert events['decision_start'][i] == events['first_pulse'][i] + t_m_steps, \
            f"Computed event timings do not align with trial length for trial {i}."
        events['trial_end'][i] = trial_length  # End of trial proper (= iti_start for sequence tasks)

        if is_sequence and 'iti_start' in batch[i]['metadata']:
            events['iti_end'][i] = int(iti_start + iti_length)

    return events


def _extract_period_boundaries(batch, task, num_trials):
    """
    Extract period boundaries for both sequence and non-sequence tasks.

    Uses actual trial-specific fixation delays (not maximum delays) to correctly
    identify where each period starts and ends. This ensures proper alignment of
    neural trajectories across trials.

    For sequence tasks: Returns 'rule_report', 'timing', 'decision', 'pre_iti', 'post_iti'
    For non-sequence tasks: Returns 'rule_report', 'timing', 'decision' (no ITI)

    Note: For sequence tasks, ITI is appended after trial data, so the data array
    for each trial includes both the trial timesteps and the following ITI timesteps.

    'pre_iti': ITI from the previous trial (trial i-1), stored in trial i-1's data
    'post_iti': ITI from the current trial (trial i), stored in trial i's data

    Args:
        batch: List of trial dicts (new batch format) with metadata including
               'initial_fixation_rule', 'initial_fixation_timing', 'iti_start', 'iti_length'
        task: Task instance
        num_trials: Number of trials

    Returns:
        dict with period names as keys, values are lists of dicts with 'start'/'end'
        For pre_iti, may also have 'trial_idx' to indicate which trial's data to extract from
    """
    # Detect task type
    is_sequence_task = task.trials_per_sequence > 1

    # Initialize period info based on task type
    if is_sequence_task:
        period_info = {
            'rule_report': [],
            'timing': [],
            'decision': [],
            'pre_iti': [],
            'post_iti': []
        }
    else:
        period_info = {
            'rule_report': [],
            'timing': [],
            'decision': []
        }

    # Get task parameters
    pulse_width_steps = int(task.pulse_width / task.dt)
    response_duration = int(task.response_period / task.dt)
    rule_report_duration = int(task.rule_report_period / task.dt)

    # Extract period boundaries for each trial
    for i in range(num_trials):
        # Get trial-specific data
        if len(batch) > 1:
            # Sequence task: N trial dicts with B=1
            total_length = batch[i]['trial_lengths'][0].item()  # Includes ITI
            t_m = batch[i]['metadata']['t_m'][0]
            initial_fix_rule = batch[i]['metadata']['initial_fixation_rule'][0]
            initial_fix_timing = batch[i]['metadata']['initial_fixation_timing'][0]
            # Get actual trial end (excludes ITI)
            if 'iti_start' in batch[i]['metadata']:
                trial_length = int(batch[i]['metadata']['iti_start'][0])
            else:
                trial_length = total_length
        else:
            # Non-sequence task: 1 trial dict with B=num_trials
            trial_length = batch[0]['trial_lengths'][i].item()
            t_m = batch[0]['metadata']['t_m'][i]
            initial_fix_rule = batch[0]['metadata']['initial_fixation_rule'][i]
            initial_fix_timing = batch[0]['metadata']['initial_fixation_timing'][i]

        # Convert to steps
        t_m_steps = int(t_m / task.dt)
        t_initial_fix_rule = int(initial_fix_rule / task.dt)
        t_initial_fix_timing = int(initial_fix_timing / task.dt)
        t_pulse = pulse_width_steps

        # Rule report period: includes initial fixation + response period
        rule_report_start = 0
        rule_report_end = t_initial_fix_rule + rule_report_duration
        rule_report_end = min(trial_length, rule_report_end)

        # Timing period: includes fixation + both pulses
        # Starts after rule report, ends after second pulse
        timing_start = rule_report_end
        timing_ready = timing_start + t_initial_fix_timing
        timing_set = timing_ready + t_m_steps
        timing_end = timing_set + t_pulse
        timing_end = min(trial_length, timing_end)

        # Decision period: last part of trial (excludes ITI)
        decision_start = trial_length - response_duration
        decision_end = trial_length

        # Add to period info
        period_info['rule_report'].append({'start': rule_report_start, 'end': rule_report_end})
        period_info['timing'].append({'start': timing_start, 'end': timing_end})
        period_info['decision'].append({'start': decision_start, 'end': decision_end})

        # ITI (only for sequence tasks)
        if is_sequence_task:
            # Pre-ITI: ITI from previous trial (stored in trial i-1's data)
            if i > 0:
                prev_iti_length = batch[i-1]['metadata']['iti_length'][0]
                prev_iti_start = batch[i-1]['metadata']['iti_start'][0]

                # Only include if previous trial has an ITI (not last in sequence)
                if prev_iti_length > 0:
                    period_info['pre_iti'].append({
                        'trial_idx': i - 1,  # Extract from previous trial's data
                        'start': int(prev_iti_start),
                        'end': int(prev_iti_start + prev_iti_length)
                    })
                else:
                    # No pre-ITI for this trial (empty period)
                    period_info['pre_iti'].append({'start': 0, 'end': 0})
            else:
                # First trial has no pre-ITI (empty period)
                period_info['pre_iti'].append({'start': 0, 'end': 0})

            # Post-ITI: ITI from current trial (stored in trial i's data)
            if len(batch) > 1:
                iti_start_in_trial = batch[i]['metadata']['iti_start'][0]
                iti_length = batch[i]['metadata']['iti_length'][0]
            else:
                iti_start_in_trial = batch[0]['metadata']['iti_start'][i]
                iti_length = batch[0]['metadata']['iti_length'][i]

            if iti_length > 0:
                period_info['post_iti'].append({
                    'start': int(iti_start_in_trial),
                    'end': int(iti_start_in_trial + iti_length)
                })
            else:
                # No post-ITI for this trial (last in sequence, empty period)
                period_info['post_iti'].append({'start': 0, 'end': 0})

    return period_info


def generate_data(task, model, num_trials=None, device='cpu'):
    """
    Generate multiple trials and collect all data.

    Handles both sequence tasks (with multiple trials per sequence and reward feedback)
    and non-sequence tasks (independent trials).

    For sequence tasks:
    - Can generate multiple sequences if num_trials > task.trials_per_sequence
    - Hidden state resets at the start of each new sequence
    - ITI periods are only included between trials within the same sequence
    - Each trial's data contains both the trial timesteps and the following ITI timesteps
      (except for the last trial in each sequence)
    - The metadata includes 'iti_start' and 'iti_length' for each trial

    Args:
        task: Task instance
        model: Trained model instance
        num_trials: Number of trials to generate (default: task.trials_per_sequence for sequence tasks, 40 otherwise)
                   For sequence tasks, can be any positive integer (will generate multiple sequences if needed)
        device: Device to run model on

    Returns:
        dict containing:
            - 'inputs': torch.Tensor of shape (num_trials, T, C) - all trial inputs (includes ITI for sequence tasks)
            - 'outputs': torch.Tensor of shape (num_trials, T, C_out) - model outputs (includes ITI)
            - 'hidden_states': torch.Tensor of shape (num_trials, H, T) - hidden states (includes ITI)
            - 'batch': list of trial dicts with metadata, inputs, targets, and masks (all include ITI for sequence tasks)
    """
    model.eval()

    # Check if this is a sequence task
    is_sequence_task = hasattr(task, 'trials_per_sequence') and task.trials_per_sequence > 1

    if is_sequence_task:
        # Sequence task: Generate trials with reward feedback
        num_trials = num_trials or task.trials_per_sequence

        # Calculate how many sequences we need to generate
        trials_per_seq = task.trials_per_sequence
        num_sequences = (num_trials + trials_per_seq - 1) // trials_per_seq  # Ceiling division

        # Generate multiple sequences and concatenate
        all_batches = []
        for seq_idx in range(num_sequences):
            batch = task.generate_batch(batch_size=1)  # List of trials_per_sequence trial dicts
            all_batches.extend(batch)

        # Truncate to exactly num_trials if we generated too many
        batch = all_batches[:num_trials]
        N = len(batch)  # Number of trials to process

        # Run trials sequentially with reward updates and ITI
        # Reset hidden state at sequence boundaries
        hidden = None
        all_outputs = []
        all_hidden_states = []
        all_inputs = []

        with torch.no_grad():
            for trial_idx in range(N):
                # Reset hidden state at the start of each new sequence
                if trial_idx % trials_per_seq == 0:
                    hidden = None
                trial_dict = batch[trial_idx]

                # Get trial data: [1, T, C]
                trial_inputs = trial_dict['inputs'].to(device)  # [1, T, 5]
                trial_targets = trial_dict['targets'].to(device)  # [1, T, 2]
                trial_eval_mask = trial_dict['eval_mask'].to(device)  # [1, T, 2]
                trial_length = trial_dict['trial_lengths'][0].item()

                # Initialize hidden state for first trial
                if hidden is None:
                    hidden = torch.zeros(1, model.hidden_size, device=device)

                # Process trial timestep by timestep
                trial_outputs_list = []
                trial_hidden_list = []

                for t in range(trial_length):
                    input_t = trial_inputs[:, t, :]  # [1, 5]
                    output_t, hidden = model(input_t, hidden)  # [1, 2], [1, H]
                    trial_outputs_list.append(output_t)
                    trial_hidden_list.append(hidden)

                # Process ITI between trials (except after last trial or at sequence boundaries)
                iti_length = 0
                is_last_in_sequence = ((trial_idx + 1) % trials_per_seq == 0)
                if trial_idx < N - 1 and not is_last_in_sequence:
                    # Stack trial outputs to compute correctness
                    trial_outputs_stacked = torch.stack(trial_outputs_list, dim=1)

                    # Check trial correctness
                    is_correct = task._evaluate_trial_correctness_batch(
                        trial_outputs_stacked,
                        trial_targets[:, :trial_length, :],
                        trial_eval_mask[:, :trial_length, :]
                    )

                    # Generate ITI inputs with reward feedback
                    iti_inputs = task._generate_iti_inputs(
                        is_correct,
                        trial_dict['metadata'],
                        task.iti_len,
                        task.reward_len
                    ).to(device)  # [1, iti_len, 5]

                    # Process ITI and save hidden states
                    iti_outputs_list = []
                    iti_hidden_list = []
                    for t in range(task.iti_len):
                        input_t = iti_inputs[:, t, :]  # [1, 5]
                        output_t, hidden = model(input_t, hidden)  # [1, 2], [1, H]
                        iti_outputs_list.append(output_t)
                        iti_hidden_list.append(hidden)

                    # Append ITI to trial data
                    trial_outputs_list.extend(iti_outputs_list)
                    trial_hidden_list.extend(iti_hidden_list)
                    iti_length = task.iti_len

                # Stack to [1, T_total, 2] and [1, T_total, H] (includes ITI if present)
                trial_outputs = torch.stack(trial_outputs_list, dim=1)
                trial_hidden_states = torch.stack(trial_hidden_list, dim=1)

                # Concatenate trial inputs with ITI inputs (zeros if no ITI)
                trial_inputs_with_iti = torch.cat([
                    trial_inputs[:, :trial_length, :],
                    iti_inputs if iti_length > 0 else torch.zeros(1, 0, 5, device=device)
                ], dim=1)

                all_inputs.append(trial_inputs_with_iti)
                all_outputs.append(trial_outputs)
                all_hidden_states.append(trial_hidden_states)

                # Add ITI start index to metadata
                batch[trial_idx]['metadata']['iti_start'] = np.array([trial_length])
                batch[trial_idx]['metadata']['iti_length'] = np.array([iti_length])

                # Update batch element to include ITI data
                # Concatenate trial data with ITI data (zeros for targets/masks during ITI)
                batch[trial_idx]['inputs'] = trial_inputs_with_iti.cpu()  # [1, T+ITI, 5]

                if iti_length > 0:
                    # Create zero targets and masks for ITI period
                    iti_targets = torch.zeros(1, iti_length, 2)
                    iti_eval_mask = torch.zeros(1, iti_length, 2)

                    batch[trial_idx]['targets'] = torch.cat([
                        trial_targets[:, :trial_length, :].cpu(),
                        iti_targets
                    ], dim=1)

                    batch[trial_idx]['eval_mask'] = torch.cat([
                        trial_eval_mask[:, :trial_length, :].cpu(),
                        iti_eval_mask
                    ], dim=1)

                    # Concatenate loss_mask if present
                    if 'loss_mask' in batch[trial_idx]:
                        iti_loss_mask = torch.zeros(1, iti_length, 2)
                        batch[trial_idx]['loss_mask'] = torch.cat([
                            trial_dict['loss_mask'][:, :trial_length, :],
                            iti_loss_mask
                        ], dim=1)
                else:
                    # No ITI - just move tensors to CPU (no concatenation needed)
                    batch[trial_idx]['targets'] = trial_targets[:, :trial_length, :].cpu()
                    batch[trial_idx]['eval_mask'] = trial_eval_mask[:, :trial_length, :].cpu()

                # Update trial length to include ITI
                total_length = trial_length + iti_length
                batch[trial_idx]['trial_lengths'] = torch.tensor([total_length], dtype=torch.long)

        # Concatenate all trials: [N, T_max, C]
        # Find max trial length
        max_trial_len = max(inp.shape[1] for inp in all_inputs)

        # Pad and stack
        inputs_padded = []
        outputs_padded = []
        hidden_states_padded = []

        for inp, out, hid in zip(all_inputs, all_outputs, all_hidden_states):
            T = inp.shape[1]
            if T < max_trial_len:
                pad_width = max_trial_len - T
                inp = torch.cat([inp, torch.zeros(1, pad_width, 5, device=device)], dim=1)
                out = torch.cat([out, torch.zeros(1, pad_width, 2, device=device)], dim=1)
                hid = torch.cat([hid, torch.zeros(1, pad_width, hidden.shape[-1], device=device)], dim=1)
            inputs_padded.append(inp)
            outputs_padded.append(out)
            hidden_states_padded.append(hid)

        # Stack and remove batch dimension: [N, T, C]
        inputs = torch.cat(inputs_padded, dim=0)  # (N, T, 5)
        outputs = torch.cat(outputs_padded, dim=0)  # (N, T, 2)
        hidden_states = torch.cat(hidden_states_padded, dim=0)  # (N, T, H)

        # Transpose hidden_states to [N, H, T] for PCA functions
        hidden_states = hidden_states.transpose(1, 2)  # (N, H, T)

        # Pad batch elements to max_trial_len (same pattern as tensor padding)
        for trial_idx in range(N):
            T = batch[trial_idx]['trial_lengths'][0].item()
            if T < max_trial_len:
                pad_width = max_trial_len - T
                # Pad inputs, targets, and masks
                batch[trial_idx]['inputs'] = torch.cat([
                    batch[trial_idx]['inputs'],
                    torch.zeros(1, pad_width, 5)
                ], dim=1)
                batch[trial_idx]['targets'] = torch.cat([
                    batch[trial_idx]['targets'],
                    torch.zeros(1, pad_width, 2)
                ], dim=1)
                batch[trial_idx]['eval_mask'] = torch.cat([
                    batch[trial_idx]['eval_mask'],
                    torch.zeros(1, pad_width, 2)
                ], dim=1)
                if 'loss_mask' in batch[trial_idx]:
                    batch[trial_idx]['loss_mask'] = torch.cat([
                        batch[trial_idx]['loss_mask'],
                        torch.zeros(1, pad_width, 2)
                    ], dim=1)

        return {
            'inputs': inputs,  # [N, T, 5]
            'outputs': outputs,  # [N, T, 2]
            'hidden_states': hidden_states,  # [N, H, T]
            'batch': batch  # List of trial dicts
        }

    else:
        # Non-sequence task: Generate independent trials
        num_trials = num_trials or 40  # Default to 40 trials

        # Generate batch with multiple independent trials
        batch = task.generate_batch(batch_size=num_trials)  # List of 1 trial dict

        assert len(batch) == 1, f"Expected 1 trial dict for non-sequence task, got {len(batch)}"
        trial_dict = batch[0]

        # Get trial data: [B, T, C]
        inputs = trial_dict['inputs'].to(device)  # [B, T, 5]
        B, T, C = inputs.shape

        # Run all trials through model timestep by timestep
        with torch.no_grad():
            # Initialize hidden state
            hidden = torch.zeros(B, model.hidden_size, device=device)

            # Process timestep by timestep
            outputs_list = []
            hidden_states_list = []

            for t in range(T):
                input_t = inputs[:, t, :]  # [B, 5]
                output_t, hidden = model(input_t, hidden)  # [B, 2], [B, H]
                outputs_list.append(output_t)
                hidden_states_list.append(hidden)

            # Stack to [B, T, C_out] and [B, T, H]
            outputs = torch.stack(outputs_list, dim=1)  # (B, T, 2)
            hidden_states = torch.stack(hidden_states_list, dim=1)  # (B, T, H)

        # Transpose hidden_states to [B, H, T] for PCA functions
        hidden_states = hidden_states.transpose(1, 2)  # (B, H, T)

        # Return data
        return {
            'inputs': inputs,  # [B, T, 5]
            'outputs': outputs,  # [B, T, 2]
            'hidden_states': hidden_states,  # [B, H, T]
            'batch': batch  # List of 1 trial dict
        }


def get_metadata(batch, outputs=None, task=None):
    """
    Extract metadata from batch for visualization.

    Args:
        batch: List of trial dicts (new batch format)
        outputs: Optional model outputs [N, T, 2] for computing decisions
        task: Optional task object (required if outputs provided)

    Returns:
        dict with metadata arrays of length N (num_trials)
    """
    # Extract metadata - handle both sequence (N dicts with B=1) and non-sequence (1 dict with B=N)
    if len(batch) > 1:
        # Sequence task: N trial dicts, each with B=1
        metadata = {}
        for key in batch[0]['metadata'].keys():
            metadata[key] = np.array([trial_dict['metadata'][key][0] for trial_dict in batch])
    else:
        # Non-sequence task: 1 trial dict with B=num_trials
        metadata = batch[0]['metadata'].copy()

    # Compute decisions if outputs provided
    if outputs is not None and task is not None:
        N = outputs.shape[0]
        decisions = []

        for i in range(N):
            if len(batch) > 1:
                trial_length = batch[i]['trial_lengths'][0].item()
            else:
                trial_length = batch[0]['trial_lengths'][i].item()

            eval_start = trial_length - int(300 / task.dt)
            decision_pred = outputs[i, eval_start:trial_length, 0].mean().item()
            decisions.append(1 if decision_pred > 0 else -1)

        metadata['decision'] = np.array(decisions)

    return metadata


def _process_single_trial(model, inputs_tensor, device='cpu'):
    """
    Helper function to process a single trial through the time-step RNN.

    Args:
        model: RNN model
        inputs_tensor: [T, C] or [1, T, C] tensor (new batch format)
        device: device to run on

    Returns:
        outputs: [1, T, C_out] tensor
    """
    # Ensure batch dimension: [T, C] -> [1, T, C]
    if inputs_tensor.ndim == 2:
        inputs_tensor = inputs_tensor.unsqueeze(0)

    B, T, C = inputs_tensor.shape

    # Initialize hidden state
    hidden = torch.zeros(B, model.hidden_size, device=device)

    # Process timestep by timestep
    outputs_list = []
    for t in range(T):
        input_t = inputs_tensor[:, t, :]  # [1, C]
        output_t, hidden = model(input_t, hidden)  # [1, C_out], [1, H]
        outputs_list.append(output_t)

    # Stack to [1, T, C_out]
    outputs = torch.stack(outputs_list, dim=1)

    return outputs


def compute_psychometric_curves(task, model, num_trials_per_interval=100, rules=None):
    """
    Compute psychometric curves showing P(Anti saccade) vs interval duration.

    Args:
        task: Task instance (should have discrete_eval=True)
        model: Trained model instance
        num_trials_per_interval: Number of trials to run per interval value
        rules: None for single-rule tasks, or list of rules (e.g., [1, -1]) for multi-rule tasks

    Returns:
        dict containing:
            - 'intervals': np.ndarray of interval values tested
            - 'accuracies': np.ndarray or dict of accuracies (if rules, dict with rule keys)
            - 'anti_probs': np.ndarray or dict of P(Anti) values
    """
    model.eval()
    intervals = task.eval_intervals

    accuracies = {rule: [] for rule in rules}
    anti_probs = {rule: [] for rule in rules}

    with torch.no_grad():
        for rule in rules:
            for t_s in intervals:
                correct = 0
                anti_count = 0
                total = 0

                for _ in range(num_trials_per_interval):
                    trial = task.generate_trial(t_s=t_s, rule=rule, has_instruction=True)
                    inputs = torch.from_numpy(trial['inputs'])  # [C, T]
                    targets = torch.from_numpy(trial['targets'])  # [C_out, T]

                    outputs = _process_single_trial(model, inputs)  # [1, T, C_out]

                    T = trial['trial_length']
                    eval_start = T - int(300 / task.dt)
                    decision_pred = outputs[0, eval_start:T, 0].mean().item()
                    decision_target = targets[0, eval_start:T].mean().item()

                    if (decision_pred > 0) == (decision_target > 0):
                        correct += 1

                    # Check if Anti choice (decision is opposite from direction)
                    is_anti = ((decision_pred > 0 and trial['stim_direction'] < 0) or
                                (decision_pred < 0 and trial['stim_direction'] > 0))
                    if is_anti:
                        anti_count += 1
                    total += 1

                accuracies[rule].append(correct / total)
                anti_probs[rule].append(anti_count / total)

    # Convert lists to numpy arrays
    for rule in rules:
        accuracies[rule] = np.array(accuracies[rule])
        anti_probs[rule] = np.array(anti_probs[rule])

    return {
        'intervals': np.array(intervals),
        'accuracies': accuracies,
        'anti_probs': anti_probs
    }
