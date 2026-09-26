//! Progress reporting and cancellation for long-running separation.
//!
//! A GUI host needs two things from a job that takes minutes: to know how
//! far it is, and to stop it. [`Control`] carries both into the processing
//! loop. Progress is counted in model windows (the unit of work), and the
//! cancellation flag is checked before every window, so a cancel takes
//! effect within one window (a few tenths of a second).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// How far a separation job is, in model windows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Progress {
    /// Windows finished so far.
    pub done: usize,
    /// Windows the job will run in total.
    pub total: usize,
}

impl Progress {
    /// Fraction finished, in `[0, 1]`. A job with no windows is finished.
    pub fn fraction(&self) -> f64 {
        if self.total == 0 {
            1.0
        } else {
            self.done as f64 / self.total as f64
        }
    }
}

/// A cancellation flag that can be shared between threads.
///
/// Cloning shares the flag: cancelling any clone cancels them all.
#[derive(Debug, Clone, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    /// A new, not cancelled token.
    pub fn new() -> Self {
        Self::default()
    }

    /// Request cancellation.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    /// Whether cancellation was requested.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

type ProgressFn = dyn Fn(Progress) + Send + Sync;

/// Progress callback and cancellation token for one job.
///
/// The callback runs on the processing thread after every window; keep it
/// cheap (store into an atomic or send on a channel).
#[derive(Clone, Default)]
pub struct Control {
    cancel: CancelToken,
    progress: Option<Arc<ProgressFn>>,
}

impl std::fmt::Debug for Control {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Control")
            .field("cancel", &self.cancel)
            .field("progress", &self.progress.is_some())
            .finish()
    }
}

impl Control {
    /// No progress callback, never cancelled.
    pub fn new() -> Self {
        Self::default()
    }

    /// Use this cancellation token.
    pub fn with_cancel(mut self, cancel: CancelToken) -> Self {
        self.cancel = cancel;
        self
    }

    /// Call `f` after every finished window.
    pub fn with_progress<F>(mut self, f: F) -> Self
    where
        F: Fn(Progress) + Send + Sync + 'static,
    {
        self.progress = Some(Arc::new(f));
        self
    }

    /// The cancellation token of this job.
    pub fn cancel_token(&self) -> &CancelToken {
        &self.cancel
    }

    /// A control for one part of a larger job: same cancellation, progress
    /// shifted by `done_before` and reported against `total`.
    pub(crate) fn part(&self, done_before: usize, total: usize) -> Control {
        let outer = self.progress.clone();
        Control {
            cancel: self.cancel.clone(),
            progress: outer.map(|f| {
                Arc::new(move |p: Progress| {
                    f(Progress {
                        done: done_before + p.done,
                        total,
                    })
                }) as Arc<ProgressFn>
            }),
        }
    }

    pub(crate) fn is_cancelled(&self) -> bool {
        self.cancel.is_cancelled()
    }

    pub(crate) fn report(&self, progress: Progress) {
        if let Some(f) = &self.progress {
            f(progress);
        }
    }
}
