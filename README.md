# Alert Processing Framework (apf)

Framework to create a dockerized pipeline to process an alert stream.

## Develop-flow

The framework is based on dockerized components with common interfaces to communicate to each other.

Every component also called `step` is a small service dockerized with a consumer and a producer, isolated from
others components.
