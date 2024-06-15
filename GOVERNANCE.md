The **WGPU project** is a set of open-source libraries that _enables application
authors to write portable and performant graphics programs_. It was originally
conceived to provide an implementation of WebGPU for Firefox as the standard
evolved, and settled into something that could be shipped on all web browsers.
WGPU has also enjoyed much contribution and use from other projects that require
graphics programming. We expect that these sorts of users will continue for the
lifetime of project, and we embrace these contributors' needs and effort as the
lifeblood of WGPU.

## Mission

The WGPU community seeks to realize the following directives through the
project: it…

1. …provides libraries for the WebGPU API that…
    1. …are correct and fully conformant.
    1. …are portable across all major platforms, that is, …
        1. …`wgpu-core` enables JavaScript platforms to implement their own
           proper WebGPU API.
        1. …`wgpu` provides a WebGPU-style API library for native applications,
           which allows shipping to all major platforms, including WebGPU's
           JavaScript API.
    1. …are performant enough to enable demanding applications.
1. …serves as a platform of experimentation for:
    1. …WebGPU standards development.
    1. …native application authors that wish to experiment with features that
       are not (yet?) standard.

## Decision-making

The WGPU community's decision-making is influenced by the following
groups:

* Community leadership:
    * Connor Fitzgerald (@cwfitzgerald)
    * Joshua Groves (@grovesNL)
    * Andreas Reich (@wumpf)
* Firefox's WebGPU team (@jimblandy, @nical, @teoxoy, @ErichDonGubler, and
  others)
* Deno's WebGPU contributors (@crowlKats)
* Other users that ship applications based on WGPU

It is no coincidence that these groups correspond to the historically most
active and consistent contributors. In general, WGPU's community structure is
meritocratic: social influence is granted proportionate to groups' contribution
to and stake in WGPU's mission.

These decision-making groups meet together regularly to discuss issues of
importance to the community, with a focus on WGPU's [mission](#Mission).

---

NOTE: The above is a snapshot of a perpetually changing state of affairs in the
WGPU community. It is not a binding contract between users and decision-makers
of the WGPU project.
