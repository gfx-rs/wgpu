fn test_misplaced_break_if() {
    loop {
        break if true;
    }
}
