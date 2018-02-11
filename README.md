# Deep_Neural_Game_Theory
It is a deep learning neural network combined with game theory.
Really, this thing is cool.

代表性著作程式: Deep Neural Game Theory深度網路賽局理論

在傳統的深度學習底下，是希望藉由Back Propagation去調整人工神經網路本身的Synapse，使得輸入端Input Neurons因此所獲得的輸出端Output Neurons，可以切進目標數值Target Neurons。
這是Hinton於1980年代發明Back Propagation最原初的目的。
這是藉由已經知道的輸入端，去推測應有的目標數值。
但是當輸入端有部分資訊是遺漏的，機器是否能夠自行【推論】應有的輸入端的數值，則是有疑問。
比方說以最微小的蚊子為例，當我們的手在他前面揮舞，蚊子接受到此一輸入端的資訊時，為何他可以自行推論出，大腦輸出飛離現場的資訊到肌肉，進而牽動自己的翅膀而飛離現場，對自己會是最佳的策略、好讓自己生存?

傳統的機器需要靠人力在輸入端輸入手部揮動的資訊，在目標數值設定飛走的資訊。所以機器以後看到手部揮動的資訊，他就會輸出飛走的資訊。

但是這個過程當中，我們已經隱存地將人類既有的觀念放到機器裡面了，就是【飛走事實上可以讓自己存活】，但是機器無須知道飛不飛走會不會對生存有所影響的資訊，也無須對此做出推理。

這樣可以說是機器擁有推理、決策能力了嗎?

為了解決這個問題，我將Back Propagation的概念更進一步昇華、應用，我是先將已經被訓練好的類神經網路，將其中部分的Input Neurons挖空，透過Back Propagation強迫這個訓練好的類神經網路去找出部分空缺的Input Neurons的最佳解答，以符合預先設定的期望的目標數值Target Neurons。但是不改變整個被訓練好的類神經網路的Synapse。

例如先訓練機器簡單的加法，例如1+1=2 ，1+2=3 ，A(Input Neurons)+B(Input Neurons)=C(Target Neurons) 。訓練完畢之後，告訴機器B=2 ，C=10
看類神經網路是否可以透過Back Propagation去找出A=8，使得整體輸出端可以切進目標數值，也就是C=10。

看似沒什麼，但是如果輸入的數值是，不飛走(Input Neurons)+手部揮舞(Input Neurons)=死亡(Target Neurons)，飛走(Input Neurons)+手部揮舞(Input Neurons)=生存(Target Neurons)，訓練完畢之後，告訴機器當人類手部揮舞時，要採取怎樣的行動才可以生存(使輸出端切進生存)，而這樣卻神奇地可以使機器推論出【飛走事實上可以讓自己存活】的最佳策略。

換而言之，Hinton的機器是為了使輸出值切進目標數值，而我的機器則是走了逆向、相反的方向，是為了使輸入值切進目標數值，使機器擁有決策推理能力，可以思考怎樣的行為(也就是輸入端)，可以符合自己的最佳利益。

那這個跟賽局理論有何關聯呢?

在賽局理論底下，每個玩家都是根據對方玩家的行動做出反應，以謀求自己利益的最大化，這個跟上面蚊子根據人類手部的行為作出反應因此飛走，以謀求自己利益的最大化是一樣的。既然上面的機器可以模擬蚊子的思考模式，推測出飛走會是最佳的策略。想當然爾，這個機器也可以模擬兩個玩家的想法，模擬兩個玩家的策略模式。

舉例而言，以最傳統的賽局理論的同時賽局(Simultaneous Game)為例:
 
如果我們把兩個玩家的策略(T, M, B)以及(L, C, R)當作是兩個輸入端，這一個人工神經網路的目標數值端則是因應上圖而對應的(0.3, 0.4)等等，將這些數值輸入訓練人工神經網路，當此一神經網路訓練完畢之後，我們可以套用前面的A(Input Neurons) + B(Input Neurons) = C(Target Neurons) 的訓練方法，只是這次A 與B 所希望達成的C 是相反，A 是希望達到(1,0) ，B 是希望達到(0,1)，訓練輸入端的方式也就是，先讓玩家column先走(先讓玩家column的輸入端透過Back Propagation以及Gradient Descent先找出切進目標數值(1, 0)的輸入值)，再讓玩家row再走(再讓玩家row的輸入端透過Back Propagation以及Gradient Descent先找出切進目標數值(0, 1)的輸入值)，最後重複以上動作。

最終機器可以自動找出納許均衡Nash Equilibrium，也就是(T, L)以及(M, C)。

想看點更暴力的嗎?

當然以上只是傳統的一般的深度網路神經Deep Feedforward Neural Network與賽局理論同時賽局Simultaneous Game的結合，但是當然這個觀念，也是可以與遞迴網路神經Recurrent Neural Network(包括LSTM、Neural Turing Machine等) 結合，這是怎樣的一個觀念呢? 詳見下述。

在賽局理論底下，除了以上的同時賽局(Simultaneous Game)以外，也包括了序列賽局(Sequential Game)，也就是大家熟悉的樹狀圖，如下所示:
 
在這序列賽局底下，各個玩家是如何揣摩對手的想法，以做出對自己最有利的決策呢?
就是採取了反向推論Back Deduction的做法，在傳統的賽局理論底下，玩家P1會去


很有趣的是，賽局理論有反向推論法Back Deduction，人工智慧有反向傳播法Back Propagation，賽局理論有同時賽局Simultaneous Game以及序列賽局Sequential Game，而人工智慧有深度網路神經DNN以及遞迴網路神經RNN，而賽局理論與人工智慧可以巧妙結合如上，發揮最大破壞力。







