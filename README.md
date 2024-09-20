CS 145 Project 7- Cell Phone Service Failures in Crowds

Author and collaborators
Author name
James Burns
Collaborators
Minghao approved my idea to use the NVIDIA Sionna library for wireless simulations.

Report Part 1: Introduction and the Power of 5G Encodings in Wireless Signal Processing
We'll be exploring Direction 2: "Pick any network outage postmortem and reproduce the similar problems based on the postmortem." I sincerely hope you enjoy grading this, the Sionna package is a powerful simulator and they claim that it's used widely in 6G research. I have no prior experience with wireless networks!

Your cell phone ceases to work in crowds, and I'm excited to explore why that grand inconvenience occurs. The network outage we'll learn about is personal: Food, water, shelter, and mobile data are the basis of a comfortable life. When we are deprived of one of those necessities, we feel thirsty, or complain about how we can't post to Instagram- equally valid concerns!

To simulate your phone's utter deprivation, and your corresponding disappointment, we'll use the Sionna Python library to explore how cell towers and phones handle demand overload. The creators define this fine library: "Sionna™ is an open-source Python library for link-level simulations of digital communication systems built on top of the open-source software library TensorFlow for machine learning". That means that she can simulate mobile data demand situations using neural networks, like a good and friendly AI- not the world conquering, human exterminating kind. At least for now.

We begin our journey with Sionna's "Hello World" notebook. If you had to navigate installing the awkward and obscure dependencies to re-run and validate my code, I apologize. That's the cost of venturing forward into the unknown- you may find yourself installing a required compiler infrastructure "LLVM" from the University of Illinois, whose icons reflect a dragon for some reason: https://llvm.org/Logo.html. This brief example shows us how Sienna uses Additive White Gaussian Noise(AWGN) to model the effects of noise and interference in a wireless signal, and the result is saved in the images folder.

Our next stop is to begin our cheerful apprenticeship with the tutorials in "Part I: Getting started with Sionna". She promises to teach us "The implementation of a point-to-point link with a 5G NR compliant code and a 3GPP channel model", so here I'll explain what that means.

A point-to-point link is communication between two endpoints/devices without any intermediate nodes or routing.
5G NR is the current generation of wireless communications technology, the NR stands for "New Radio"
A 3GPP "3rd Generation Partnership Project" channel model describes a standardized propagation of wireless signals in varied environments.
So we've got our first taste here of a "phone" connecting to a "cell tower". Progress! The steps involved in transferring the data in this example are:

Data is created from a binary source (sn.utils.BinarySource()).
We create a Quadrature Amplitude Modulation (QAM) constellation, which graphs 16 points from 4 bits per symbol (16-QAM_Constellation.png). These 16 points graphically represent different combinations of amplitude and phase of the radio waves transmitting the data.
A mapper maps the information as a series of bits to the combinations of amplitude and phase states. For example, 0010 is represented by an exact waveform pattern. The mapper class implements a Keras layer.
The AWGN channel (sn.channel.AWGN()) adds white noise to the signals for our simulation, which represents interference from other sources.
a demapper completes the inverse operation of the mapper, processing to remove channel distortion, then converts the symbols back into binary information. The math here computes log-likelihood ratios (LLRs) from received noisy samples.
This experiment yielded the following results:

First 8 transmitted bits: [1. 0. 0. 1. 0. 0. 0. 1.]
First 2 transmitted symbols: [-0.32+0.95j 0.32+0.95j]
First 2 received symbols: [-0.37+0.8j 0.3 +0.98j]
First 8 demapped llrs: [ 18.84 -49.28 -13.16 8.64 -15.27 -67.52 -16.73 17.76]
In plain English, this means that the received symbols were very close to the sent ones, and the receiver would be able to correctly identify the bits transferred!

More vocabulary for you: The Energy per Bit to Noise power Spectral Density ratio(EbNo) in our model above was static at 10dB. This matters below.

Our next example does the same job, but wraps the Sionna-based communication system into a Keras model, without any encoding. It's also substantially scaled up. No longer using a static EbNo of 10 dB, we now cycle through a range of -3.0 to 5.0 dB. Success is measured by the Bit Error Rate(BER), which is a proportion out of one of how many bits arrived with errors.

Results:

The best and worst results were found at the end of the dB spectrum used. at -3.0, 21% of bits were erroneous, compared to at 5.0 where only 4.18% of bits were incorrect.
Moving on, results from a similar model that used encodings (5G compliant low-density parity check codes and Polar codes) reduced the error rate to 3.21% at -3.0 dB. In fact, at 5.0 db the BER was 0 for 30.72 million transferred bits. These encodings are where a power of 5G technology shines, in providing a nearly perfect transmission of data despite signal interference.

Report Part 2: Multi Input Multi Output Simulation With 1 User Terminal "Cell Phone" and 1 Base Station "Cell Tower"
In "Sionna_tutorial_part_3" the default values are 1 UT and 1 BS, representing a cell phone and a cell tower. A key new feature here is the OFDMChannel layer, which implements Orthogonal Frequency Division Multiplexing. OFDM is used in 5G cell networks to encode data on multiple carrier frequencies. Viewing this tutorial helped me understand more pieces of the puzzle before tackling my main concern: Simulating poor cell service caused by excess demand.

Report Part 3: The Real Deal: MIMO Simulation With Varying Number of User Terminals "Cell Phones" and 1 Base Station "Cell Tower"
In "Realistic_Multiuser_MIMO_OFDM_Simulations", I finally get to simulate what we've been waiting for: multiple cell phones communicating with a single tower. The base case has 4 UT's communicating with 1 BS. This simulation uses much of the basic elements we referred to in part 1, but uses many more features like StreamManagement, ResourceGrids, Antennas, and an LDPC5GEncoder/Decoder. But the most important metric we want to see is our old friend, the Bit Error Rate(BER). In the default scenario, the BER is 0.006%, and the network is operating smoothly!

At 4 UT's and 1 BS: BER = 0.006%
At 8 UT's and 1 BS: BER = 1.25%
At 16 UT's and 1 BS: BER = 21.47%
At 32 UT's and 1 BS: BER = 35.88%
At 64 UT's and 1 BS: BER = 41.87%
At 128 UT's and 1 BS: BER = 45.67%
The BER required for 4G/5G networks to reliably transfer data is between 1x10^-6 and 1x10^-9. Applied to the case above, Cell service would be fine at 4 or fewer users, frustratingly slow at 8 users, and completely unusable at 16 users and above.

What if we add more Base Stations/"Cell Towers"?

At 4 UT's and 2 BS: BER = 0%
At 8 UT's and 2 BS: BER = 0.6%
At 16 UT's and 2 BS: BER = 15.4%
At 32 UT's and 2 BS: BER = 30.67%
Cell service with 2 BS becomes completely unusable at around 16 users in the cell sector.

The technical reason for cell phone service failure in crowds is that the BER increases as the network cannot process all signals through the interference. Imagine a calm pool where if you put your feet in the water, it creates a distinguishable wave. Now imagine a pool full of 16 teenagers playing around. The "signal" wave from you putting your feet in is immediately lost in all the other waves. Now we know! Thanks for reading!

Citations
https://www.youtube.com/watch?v=0faCad2kKeg Great video that helped me understand basic wireless concepts https://arxiv.org/pdf/2303.11103.pdf Research paper promoting Sionna package features https://info.support.huawei.com/info-finder/encyclopedia/en/QAM.html 16-QAM Information https://www.fiberoptics4sale.com/blogs/archive-posts/95047174-what-is-ber-bit-error-ratio-and-bert-bit-error-ratio-tester BER Info https://developer.nvidia.com/sionna https://github.com/nvlabs/sionna https://nvlabs.github.io/sionna/examples/Hello_World.html https://llvm.org/ Compiler https://www.reddit.com/r/explainlikeimfive/comments/dl7ve0/eli5_when_inside_a_large_group_of_people_at_a/ Initial research https://www.latimes.com/entertainment-arts/music/story/2022-03-10/mitski-bruno-mars-silk-sonic-cell-phones-concerts Cell phones at concerts research https://www.simplypsychology.org/maslow.html Where does mobile data fit in Maslow's hierarchy of needs? https://chat.openai.com For many definitions of terms

Grading notes (if any)
Extra credit attempted (if any)
