# runway_for_ml


# Project Statement

We recognize that despite the emergence of deep Learning frameworks such as `pytorch` and higher-level frameworks such as `pytorch lightning` that separates the concerns of data, model, training, inference, and testing. There are still needs for yet a higher-level framework that addresses **data preparation**, **experiments configuration**, **systematic logging**, and **working with remote GPU clusters** (e.g. HPC). These common, indispensable functionalities are often re-implemented for individual projects, which hurt reproducibility, costs precious time, and hinders effective communications between ML developers.

We introduce **Runway**, a ML framework that delivers the last-mile solution so that researchers and engineers can focus on the essentials. In particular, we aim to 
1. Provide a **configurable data processing pipeline** that is easy to inspect and manipulate.
2. Provide an **experiment configuration system** so that it is convenient to conduct experiments in different settings without changing the code.
3. Provide a **systematic logging system** that makes it easy to log results and manage experiments both locally or on online platforms (e.g. weights-and-bias)
4. Provide a set of **tools that simplifies training/testing on remote GPU clusters** (e.g. HPC/Multiple GPU training)

With *Runway*, we hope to help ML researchers and engineers focus on the essential part of machine learning - data processing, modeling, inference, training, and evaluation. Our goal is to build a robust and flexible framework that gives developers complete freedom in these essential parts, while removing the tedious book-keeping. The philosophy, if taken to the extreme, entails that every line of code should reflect a design decision, otherwise there should be a configuration or a ready-to-use tool available. 