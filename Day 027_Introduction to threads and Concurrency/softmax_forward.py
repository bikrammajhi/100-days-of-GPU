## Adapated From https://blog.nagi.fun/triton-intro-softmax?locale=en ##

@triton.jit
def _softmax_fwd_kernel(
    output_ptr,
    stride_output_row,
    input_ptr,
    stride_input_row,
    num_cols,
    block_size: tl.constexpr,
):
    # Get the ID of the current program (row index)
    row_index = tl.program_id(0)

    # Calculate the starting pointer for the current row
    row_start_ptr = input_ptr + (row_index * stride_input_row)
    col_offsets = tl.arange(0, block_size)
    input_pointers = row_start_ptr + col_offsets

    # Create a mask to prevent out-of-bounds access
    row_mask = col_offsets < num_cols

    # Load data from global memory to shared SRAM
    row = tl.load(input_pointers, mask=row_mask, other=float("-inf"))

    # Softmax calculation
    safe_row = row - tl.max(row, axis=0)
    numerator = tl.exp(safe_row)
    denominator = tl.sum(numerator, axis=0)
    sm_out = numerator / denominator

    # Write results back to global memory
    output_row_ptr = output_ptr + (row_index * stride_output_row)
    output_pointers = output_row_ptr + col_offsets
    tl.store(output_pointers, sm_out, mask=row_mask)
