# Merge Queue

We currently use the [Mergify](https://mergify.com) merge queue to handle merging PRs in.
We are using this instead of the GitHub merge queue feature because it supports choosing
squash-and-merge and rebase-and-merge on a per-PR basis while the native queue requires
choosing on a repository level. Because we use rebase-and-merge quite often in our flows,
we didn't want to give that up.

## How to Merge

You won't be able to merge things normally while this is active. You _will_ see a "enable auto-merge"
button, but the requirements will never be satisfied for this.

In order to kick off a merge, anyone with write perms can write `@mergifyio queue` at the beginning of a comment.
This needs to be a normal comment not in a review. This will add the PR as a candidate for merging.

The PR will not be added to the merge queue properly until all of the merge requirements are satisfied.
Once the merge requirements are satisfied, the PR will automatically be added to the queue, and will be
given the `queue: queued` label by Mergify. If Mergify has replied :+1: to the comment, which it does
practically instantly, you know that it has seen it.

Once the PR is added to the queue, one of two things happen:

- If the PR is already up to date (meaning it incorporates changes from trunk) AND there are no PRs currently
  in the merge queue, the PR will be directly merged by Mergify. Because it's fully up to date, PR CI is enough
  proof that it will pass CI on trunk.
- Otherwise Mergify will create a draft pr with the prefix "merge queue:". This draft PR will contain the PR
  merged with the current "top" of the merge queue. Once all checks on this draft PR have passed and all PRs ahead
  of the PR on the queue have merged, it will merge the original _PR_ with the method you specified.

If Mergify removes a PR from the queue, or to check on the status on the queue, inspect the "Mergify Merge Queue"
check on the PR for the reason. The PR will have the `queue: dequeued` label added to it in this case. You can
re-queue it when it's ready to go again `@mergifyio queue`.

If you would like to stop the process before the PR has been merged, you can call `@mergifyio dequeue`. It will then
take all the necessary steps to remove the PR from the queue, and will not automatically queue it when merge requirements
are met.

### Specifying Merge Methods

By default, Mergify is configured to squash-and-merge all PRs. If a PR is viable to rebase-and-merge, please add
the label `merge: rebase` before you add it to the queue. Mergify will then rebase-and-merge the PR.

### Old PRs

This problem will resolve itself in time, but PRs that were _updated_ before Mergify was set up need a call to
`@mergifyio refresh` for their merge requirements to be properly checked. I (cwfitzgerald) suspect that this
will need to happen whenever we significantly change Mergify configuration, but that is still to be seen.

### Admin Overrides

Repository Admins (@jimb and @cwfitzgerald) are able to override this to merge something without the queue.
We will not do this without exigent circumstances and will document our reason for overriding
the normal system.

### PR Spam

Because it creates a new draft PR every time it needs to test an out-of-date PR there is a decent amount of PR
spam when the queue is actively merging things. I don't have a good solution for those who get notifications
on every PR. If, however, you get emails on every PR, you can filter out anything from Mergify, as those are
the only PRs that it will generate, so you're safe to ignore them all.

## Dependencies

You can express time and PR based dependencies.

If you add `Depends-On: #XXXX` to the PR body, the queue will not add the PR to the queue until _after_
the mentioned PR is also added to the queue and will not merge it until after the predecessor one gets merged.

Similarly, if you add `Merge-After: <timestamp>` to the PR body, you can cause a PR to wait to be queued until
after a certain time. Useful if you want to wait for an author's response for a certain time, but not hold the
PR up indefinitely.

See https://docs.mergify.com/merge-protections/builtin for more information on these.

## Auditing The Queue

To view a history of the queue, going to the [Mergify Dashboard](https://dashboard.mergify.com/orgs/gfx-rs/home) and
logging in with OAuth, you should be able to see the history of the merge queue, all actions taken with Mergify.

#### On Github

Because of how CI jobs are assigned to commits, you won't see the full CI for queue-merged CI when you view each commit,
you'll see the few jobs that run specifically on trunk (publishing docs, etc). If you want to see the CI state or logs
for the merge, go to the PR that landed that commit. Then:

- If it landed directly, look at the CI results for the PR.
- If it needed to be updated, the draft PR that ran the merge queue checks will be in the PR timeline. Click that, then
  go to the "checks" tab and you can see all the tests that caused the change to land.

#### On Mergify

If you go to "Merge Queue" -> "Status" -> "Merged" -> Click on a PR it will show you the history of that particular PR.
This will include all of the checks that were run to make sure the PR was clean. You can click on each one to be brought
straight to that Job's log

## References

- All Mergify commands: https://docs.mergify.com/commands/
- Lifecycle of the merge queue: https://docs.mergify.com/merge-queue/lifecycle/
- Chrome Extension: https://chromewebstore.google.com/detail/mergify/idhdcccjlcijifdphaicgnmhifpmilge
- Firefox Extension: https://addons.mozilla.org/en-US/firefox/addon/mergify/

Both extensions are shortcuts to add comments with the relevant commands or go to the relevant mergify page.
