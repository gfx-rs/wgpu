/// If the first iterator is longer than the second, the zip implementation
/// in the standard library will still advance the the first iterator before
/// realizing that the second iterator has finished.
///
/// This implementation will advance the shorter iterator first avoiding
/// the issue above.
///
/// If you can guarantee that the first iterator is always shorter than the
/// second, you should use the zip impl in stdlib.
pub(crate) struct ZipWithProperAdvance<
    A: ExactSizeIterator<Item = IA>,
    B: ExactSizeIterator<Item = IB>,
    IA,
    IB,
> {
    a: A,
    b: B,
    iter_a_first: bool,
}

impl<A: ExactSizeIterator<Item = IA>, B: ExactSizeIterator<Item = IB>, IA, IB>
    ZipWithProperAdvance<A, B, IA, IB>
{
    pub(crate) fn new(a: A, b: B) -> Self {
        let iter_a_first = a.len() <= b.len();
        Self { a, b, iter_a_first }
    }
}

impl<A: ExactSizeIterator<Item = IA>, B: ExactSizeIterator<Item = IB>, IA, IB> Iterator
    for ZipWithProperAdvance<A, B, IA, IB>
{
    type Item = (IA, IB);

    fn next(&mut self) -> Option<Self::Item> {
        if self.iter_a_first {
            let a = self.a.next()?;
            let b = self.b.next()?;
            Some((a, b))
        } else {
            let b = self.b.next()?;
            let a = self.a.next()?;
            Some((a, b))
        }
    }
}

impl<A: ExactSizeIterator<Item = IA>, B: ExactSizeIterator<Item = IB>, IA, IB> ExactSizeIterator
    for ZipWithProperAdvance<A, B, IA, IB>
{
    fn len(&self) -> usize {
        self.a.len().min(self.b.len())
    }
}
