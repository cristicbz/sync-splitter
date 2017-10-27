[![Build Status](https://travis-ci.org/cristicbz/sync-splitter.svg?branch=master)](https://travis-ci.org/cristicbz/sync-splitter)
[![Docs](https://docs.rs/sync-splitter/badge.svg)](https://docs.rs/sync_splitter)

SyncSplitter
===

A `SyncSplitter` allows multiple threads to split a mutable slice at the same time.

It's a bit like a `Sync` arena, where you can 'allocate' elements in the same
`Vec` (or some other mutable slice) in parallel, then at the end have get back
the elements.

You kinda need this sort of thing to build trees or graphs in parallel (eg. with
[rayon](https://github.com/rayon-rs/rayon)) without allocating each node
individually. The motivating case was a
[BVH](https://en.wikipedia.org/wiki/Bounding_volume_hierarchy) implementation.


Example
===
```rust
use sync_splitter::SyncSplitter;

// We'll build a binary tree and store it in an array where each node points to its first,
// child, with the two children always adjacent.
#[derive(Default, Copy, Clone)]
struct Node {
    // We'll store the depth in-lieu of some actual data.
    height: u32,

    // The index of the first child if not a leaf. The second will be `first_child_index + 1`.
    first_child_index: Option<usize>,
}

fn create_children(parent: &mut Node, splitter: &SyncSplitter<Node>, height: u32) {
    if height == 0 {
        return;
    }

    // Calling `pop_two` (or `pop_n`) gets two consecutive elements from the original slice.
    let ((left, right), first_child_index) = splitter.pop_two().expect("arena too small");
    *parent = Node {
        height,
        first_child_index: Some(first_child_index),
    };
    rayon::join(|| create_children(left, splitter, height - 1),
                || create_children(right, splitter, height - 1))
}

let mut arena = vec![Node::default(); 500];
let num_nodes = {
    let splitter = SyncSplitter::new(&mut arena);
    {
        let (root, _) = splitter.pop().expect("arena too small");
        create_children(root, &splitter, 5);
    }
    splitter.done()
};
assert_eq!(num_nodes, 63);
arena.truncate(num_nodes);

// `arena` now contains all the nodes in our binary tree.

```
