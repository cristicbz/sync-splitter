//! A `SyncSplitter` allows multiple threads to split a mutable slice at the same time.
//!
//! See the module docs for more information.
//!
//! This is handy when you're building a tree (or some other graph) with multiple threads and you
//! want all the node to live in the same array once built, like a `Sync` arena allocator.
//!
//! Example
//! ===
//! ```rust
//! # mod rayon {
//! #   pub fn join<A: FnOnce(), B: FnOnce()>(a: A, b: B) {
//! #       a();
//! #       b();
//! #   }
//! # }
//! use sync_splitter::SyncSplitter;
//!
//! // We'll build a binary tree and store it in an array where each node points to its first,
//! // child, with the two children always adjacent.
//! #[derive(Default, Copy, Clone)]
//! struct Node {
//!     // We'll store the depth in-lieu of some actual data.
//!     height: u32,
//!
//!     // The index of the first child if not a leaf. The second will be `first_child_index + 1`.
//!     first_child_index: Option<usize>,
//! }
//!
//! fn create_children(parent: &mut Node, splitter: &SyncSplitter<Node>, height: u32) {
//!     if height == 0 {
//!         return;
//!     }
//!
//!     // Calling `pop_two` (or `pop_n`) gets two consecutive elements from the original slice.
//!     let ((left, right), first_child_index) = splitter.pop_two().expect("arena too small");
//!     *parent = Node {
//!         height,
//!         first_child_index: Some(first_child_index),
//!     };
//!     rayon::join(|| create_children(left, splitter, height - 1),
//!                 || create_children(right, splitter, height - 1))
//! }
//!
//! let mut arena = vec![Node::default(); 500];
//! let num_nodes = {
//!     let splitter = SyncSplitter::new(&mut arena);
//!     {
//!         let (root, _) = splitter.pop().expect("arena too small");
//!         create_children(root, &splitter, 5);
//!     }
//!     splitter.done()
//! };
//! assert_eq!(num_nodes, 63);
//! arena.truncate(num_nodes);
//!
//! // `arena` now contains all the nodes in our binary tree.
//!
//! ```

use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::slice;

/// A `SyncSplitter` allows multiple threads to split a mutable slice at the same time.
///
/// See the module docs for more information.
pub struct SyncSplitter<'a, T: 'a + Sync> {
    data: *mut T,
    len: usize,
    next: AtomicUsize,
    dummy: PhantomData<&'a mut [T]>,
}

impl<'a, T: 'a + Sync> SyncSplitter<'a, T> {
    /// Creates a new `SyncSplitter` from a slice.
    ///
    /// Panics
    /// ===
    ///
    /// If `slice.len() >= isize::MAX`.
    pub fn new(slice: &'a mut [T]) -> Self {
        assert!(slice.len() <= isize::max_value() as usize);
        SyncSplitter {
            data: slice.as_mut_ptr(),
            len: slice.len(),
            next: AtomicUsize::new(0),
            dummy: PhantomData,
        }
    }

    /// Pops one mutable reference off the slice and returns it.
    ///
    /// Also returns the element's index in the original slice.
    ///
    /// Returns `None` if the underlying slice was exhausted. After that, all future `pop` calls
    /// will return `None`.
    #[inline]
    pub fn pop(&self) -> Option<(&mut T, usize)> {
        self.bump(1).map(|index| {
            (unsafe { &mut *self.data.offset(index as isize) }, index)
        })
    }

    /// Pops two mutable references off the slice and returns them.
    ///
    /// Also return the returned slice's offset into the original slice.
    ///
    /// Returns `None` if the underlying slice had one or zero elements left. Note that this
    /// exhausts the slice even when only one element was left (and said element is not returned).
    #[inline]
    pub fn pop_two(&self) -> Option<((&mut T, &mut T), usize)> {
        self.bump(2).map(|index| {
            (
                unsafe {
                    (
                        &mut *self.data.offset(index as isize),
                        &mut *self.data.offset(index as isize + 1),
                    )
                },
                index,
            )
        })
    }

    /// Pops a mutable slice of a given length and returns it.
    ///
    /// Also return the returned slice's offset into the original slice.
    ///
    /// Returns `None` if not enough elements were left in the underlying slice. Note that this
    /// exhausts the slice even though no elements are returned. Any further calls to `pop` will
    /// return `None`.
    #[inline]
    pub fn pop_n(&self, len: usize) -> Option<(&mut [T], usize)> {
        self.bump(len).map(|index| {
            (
                unsafe { slice::from_raw_parts_mut(self.data.offset(index as isize), len) },
                index,
            )
        })
    }


    /// Consumes the splitter and returns the total number of popped elements.
    #[inline]
    pub fn done(self) -> usize {
        self.next.load(Ordering::SeqCst)
    }


    fn bump(&self, len: usize) -> Option<usize> {
        assert!(isize::max_value() as usize - self.len >= len);
        let index = self.next.fetch_add(len, Ordering::SeqCst);
        if len < self.len && index <= self.len - len {
            Some(index)
        } else {
            self.next.store(self.len, Ordering::SeqCst);
            None
        }
    }
}

unsafe impl<'a, T: Sync> Sync for SyncSplitter<'a, T> {}

#[cfg(test)]
mod tests {
    use super::SyncSplitter;
    use std::isize;

    #[test]
    fn basics() {
        let mut buffer = [1u32, 2, 3, 4, 5];
        let splitter = SyncSplitter::new(&mut buffer);

        assert_eq!(splitter.pop_n(0), Some((&mut [][..], 0)));
        assert_eq!(splitter.pop_n(1), Some((&mut [1u32][..], 0)));
        assert_eq!(splitter.pop(), Some((&mut 2u32, 1)));
        assert_eq!(splitter.pop_n(2), Some((&mut [3u32, 4u32][..], 2)));
        assert_eq!(splitter.pop_n(1), Some((&mut [5u32][..], 4)));
        assert_eq!(splitter.done(), 5);
    }

    #[test]
    fn runs_out() {
        let mut buffer = [1u32, 2, 3, 4, 5];
        let splitter = SyncSplitter::new(&mut buffer);

        splitter.pop_n(3);
        assert_eq!(splitter.pop_n(3), None);
        assert_eq!(splitter.pop_n(1), None);
        assert_eq!(splitter.pop(), None);
        assert_eq!(splitter.done(), 5);
    }

    #[test]
    fn reads_writes() {
        let mut buffer = [1u32, 2, 3, 4, 5, 6];
        {
            let splitter = SyncSplitter::new(&mut buffer);
            {
                let (one_to_three, _) = splitter.pop_n(3).unwrap();
                let (four, _) = splitter.pop().unwrap();
                let (five, _) = splitter.pop_n(1).unwrap();

                one_to_three[0] = 100;
                one_to_three[1] = 200;
                one_to_three[2] = 300;

                *four = 400;
                five[0] = 500;
            }
            splitter.done();
        }

        assert_eq!(buffer, [100u32, 200u32, 300u32, 400u32, 500u32, 6]);
    }

    #[test]
    fn check_len_underflow() {
        let mut buffer = [1u32, 2, 3, 4, 5];
        let splitter = SyncSplitter::new(&mut buffer);

        splitter.pop_n(3);
        assert_eq!(splitter.pop_n(100), None);
        assert_eq!(splitter.pop_n(1), None);
        assert_eq!(splitter.pop(), None);
        assert_eq!(splitter.done(), 5);
    }

    #[test]
    #[should_panic]
    fn check_next_overflow() {
        let mut buffer = [(); isize::MAX as usize];
        let splitter = SyncSplitter::new(&mut buffer);
        splitter.pop();
    }

    #[test]
    fn check_isize_max_ok() {
        let mut buffer = [(); isize::MAX as usize];
        let _splitter = SyncSplitter::new(&mut buffer);
    }

    #[test]
    fn check_isize_max_minus_one_pop_ok() {
        let mut buffer = [(); isize::MAX as usize - 1];
        let splitter = SyncSplitter::new(&mut buffer);
        splitter.pop();
    }
}
