### 大语言模型时代的到来
ChatGPT 的出现，把整个世界带到了语言模型时代，我们看到了LLMs在自然语言处理（NLP）、计算机视觉（CV）、生成式任务、基本推理等多方面的能力。LLMs可以执行多种不同的任务，例如问答、解题、coding，翻译，撰文，写SQL等等。很多任务的完成度开始匹配甚至有超越人类的趋势，未来LLMs可能颠覆人们使用搜索引擎，内容创作，编写代码等的方式。
拥抱LLMs已然是大势所趋，对于我们来讲，如何构建特定领域或者垂直领域的LLM已成为最迫切的问题。
![image](https://github.com/hellogxp/LLMs/assets/1506016/1a1d96f6-356f-497b-a575-e697da82a6c7)

### 什么是大模型
何为大模型，大模型通常是指在机器学习中具有大量参数和复杂结构的模型。这些模型通常由多层神经网络组成，每层包含大量的神经元和连接，实际上目前我们目前提到较多的是Large Language Model也就是LLM。实际上在学术界对于大模型并没有一个统一的定义，在这里我们不阐述概念，只是讲一下大模型应该有的两个特点。
首先，它的规模要足够大，包括参数和数据规模，这个很容易理解，不去深究。
第二，它要是生成式的（Generative），何为生成式呢，我们要顺带提一下判别式（Discriminative）。
#### 生成模型 VS 判别模型
通俗来讲，生成式模型可以深入彻底地学习事物，判别式可以学习仅识别它所看到的事物之间的差异。例如，生成式模型可以学习板材的特征，如颜色、纹理、形状等，而判别式只学习有助于轻松对它们进行分类的特征，例如颜色。
生成式模型学习联合概率分布 $p(x,y)$，例如，有输入数据 x 并且希望将其分类为 y 标签。由于生成模型分别对每个类别进行建模，因此可以在单个类别的基础上进行训练。一些示例包括朴素贝叶斯、隐马尔可夫模型 (HMM)、马尔可夫随机场。
判别模型使用贝叶斯定理学习条件概率分布 $p(x|y)$，主要目的是区分类别，需要在包含所有类别示例的单个训练集上进行训练。一些判别模型包括Conditional Random Fields (CRFs)、Logistic regression、Traditional neural networks和Nearest neighbour.。这些模型比生成模型更复杂。
生成算法对数据的生成方式进行建模，以便对信号进行分类，而判别模型则不关心数据的生成方式，它只是对给定信号进行分类。    
![image](https://github.com/hellogxp/LLMs/assets/1506016/ac157428-2086-40cc-9647-a8f708258bfa)    
举个通俗点的例子，我们给A和B两个同学分配了一个任务，区分狗和猫。A同学善于学习，整理猫和狗的各种数据进行建模，深入总结归纳，最终形成了一套完整的方法论，A同学通过自己这套推理逻辑可以比较准确的区分区猫还是狗。B同学，只是简单记一些狗和猫的特征，比如狗体长较长，猫腿较短等，B最终只是靠这些特征记忆来区分狗和猫，如果给他的任务中有他见过的狗或者猫，那么他可以很轻松的区分，否则，就难以胜任了。
![image](https://github.com/hellogxp/LLMs/assets/1506016/3a56d044-51e8-48dc-91ee-9be801962d80)

#### 生成式模型和判别式模型的有优劣
* 生成式模型
  * Pros：擅长无监督机器学习
  * Cons：计算成本非常昂贵
* 判别式模型
  * Pros：擅长监督机器学习
  * Cons：计算成本较低

### 大模型的问题
随着大模型应用落地需求的不断增加，其问题也开始凸显出来。
1. 幻觉问题（Hallucinations），LLM 幻觉是指大型语言模型 (LLM) 生成不正确、无意义或不真实的文本。 受限于LLM的培训数据，当他们遇到不属于训练数据集的查询，或者当他们的训练数据集包含错误信息时，会产生幻觉，也就是会乱讲。
2. 干预困难，深度学习，尤其大模型，拥有巨大的参数量和模型结构，模型的输出，对于人类来讲是个黑盒，想要修改其中部分数据又不破坏LLMs原有结构和能力及其困难。
3. 数据需求高：大模型通常需要大量数据进行训练，以获得良好的性能。这可能对数据收集和存储造成挑战，尤其是对于一些特定领域或任务而言，难以获得足够的高质量数据。
4. 输出不稳定，大模型的输出可能会受到随机因素的影响，导致相同的输入在不同运行时产生不同的输出。这是由于模型中存在随机初始化、随机采样或随机正则化等操作，使得模型的输出结果具有一定的随机性。
5. 泛化能力不足：大模型在大规模数据集上的性能可能很好，但在小规模或特定领域的数据上可能表现不佳。这是因为大模型倾向于过度拟合训练数据，而对于未见过的数据缺乏泛化能力。
6. 大模型训练需要巨量算力，pre-training和fine tuning成本极高，包括时间和金钱。直接使用大模型成本很低，但必须认识到它有时弱智，希望它百分之百准确非常困难的。
7. 隐私和安全风险：大模型通常需要大量的个人数据进行训练，可能涉及到隐私和安全风险。此外，大模型也可能被用于攻击和滥用，例如生成虚假信息或进行隐秘的个人身份识别。

这里我们重点讲一下幻觉和泛化能力不足的问题，原因之一是LLMs在学习LLMsLong-Tail Knowledge的局限性，这篇paper《Large Language Models Struggle to Learn Long-Tail Knowledge》中提到，语言模型可以学习互联网上的丰富知识，但是有的信息在互联网上却鲜有出现。大模型回答基于事实问题的能力取决于在预训练期间接收到了多少与此问题相关的文档，也就是说，LLMs回答问题的准确性与相关性文档数量存在极强的相关性和因果关系。    
![image](https://github.com/hellogxp/LLMs/assets/1506016/f86e746a-33ea-4dd2-83be-6f136f665959)    
从上图可以看到，当相关性文档数从$10^{1}$增加到$10^{4}$的时候， BLOOM-176B的准确率将从5%跃升到55%。
我们可以做一个反向实验，将与问题相关的文档从LM中删除，然后从新训练模型，然后对比原始模型与从新训练过模型的准确性，差异极大。    
![image](https://github.com/hellogxp/LLMs/assets/1506016/297c4f51-e9be-4c58-9f4c-8a01ab65a896)    
正反两方面来验证，说明了，LLMs对于中长尾知识的正确回答能力取决于它在预训练阶段是否被灌入了对应的知识以及知识的量级。对于长尾知识，并不是说针对性的加入一条相关性知识，LM就能学习并且记忆，可能需要输入 $10^N$
，这些知识才能隐式的存储在其参数中，也就是被记忆。
这里有两个问题，第一，这些中长尾知识获取成本极高，甚至会涉及隐私，安全等问题，比如企业数据，商品销售数据，细分行业专业知识，较新的技术文档等。第二，要获取大量的长尾知识才可能满足训练条件。由此得出结论，LLMs在学习长尾知识方面是短板，想在预训练阶段加入长尾知识难以实现。

### 影响大模型的手段
既然大语言模型在中长尾的问答准确率依赖预训练数据中相关文档的数量，我们提出几种干预方法：
#### 1. 扩展数据规模
目前的LLMs，动辄数千亿token，为了一些领域知识，为了一些普遍性不强的领域知识来添加大量的数据，成本极高，效果也并不好。再就是增加预训练数据的多样性，但实验表明，这带来的首页也微乎其微，因为很多数据源及其相关，尤其是公开的互联网数据。比如我们看这四个数据集The Pile、ROOTS、C4、OpenWebText 和 Wikipedia的Spearman相关系数。也就是说这四种数据源对于长尾问答的贡献基本一致，并没有因为多样性给模型带来更多长尾记忆。    
![image](https://github.com/hellogxp/LLMs/assets/1506016/6ab5d46f-32e2-4426-bfca-a086dfd5c5a4)
#### 2. 模型缩放
扩大模型规模确实可以带来更好的QA性能，但是这意味着指数级别参数的增加，随之带来高昂的训练成本。基于BLOOM模型，需要超过$10^{18}$甚至$10^{20}$才能匹配监督基线或者达到人类基本表现。    
![image](https://github.com/hellogxp/LLMs/assets/1506016/36bee3e1-7370-4c95-8e53-f74d9228ca49)    
另一种模型缩放的方法是，修改训练目标，我们可以增加训练周期数，可以增加epoch同时减少数据size，让模型尽可能多的记忆，或者可以考虑修改训练损失以最大限度减少遗忘。
#### 3. 增加辅助检索模块
使用检索增强 Retrieval Augmentation，目前我们使用的LM基本上都是一个与外部系统隔离的系统，对于知识密集型的任务，自然想到对LMs进行检索增强，也就是将LMs和检索增强模块结合，Retrieval Augmentation模块返回相关上下文，以减少对预训练数据中相关知识的依赖，提高准确性。
前两种方法的实时训练成本较高，同时知识也会过期，相比之下，检索增强可以实时传递知识给模型，成本低级。大模型微调通常是把问答知识对finetune到模型中，可以改变模型的概率分布，调提升zero-shot的效果，但模型是否使用这些知识不可控，需要大量的标注数据，这也是一项繁重任务，繁琐耗时。相比而言，检索增强控件更可控。    
![image](https://github.com/hellogxp/LLMs/assets/1506016/1d8401e3-db4d-4be0-8543-2aa354c0d21a)

### 什么是 RAG
经过前面的铺垫，实际上我们对于Retrieval Augmented已经有了一定的认识。Retrieval Augmented Generation（RAG）是一个人工智能框架，可以从外部知识库检索事实，使得LLMs无需重新训练或者微调即可利用额外的数据资源，增强了其对上下文的理解，帮助预先训练的大语言模型生成更准确、且最新的信息并减少幻觉。
检索增强生成能够克服传统文本生成模型的一些限制而广受关注。 虽然 GPT 这样的生成式模型在生成连贯且上下文相关的文本方面表现极佳，但在特定的事实信息或需要高度专业知识的任务也就是知识密集型任务（Knowledge-Intensive NLP Tasks）中，表现往往差强人意。 通过结合检索和生成的优势，RAG 模型解决了这些限制，并为更通用、更有效的文本生成方向带来了研究希望。
从chatbots和content generation到QAs和language translation，RAG提供了合理的解决方案来提高应用程序的质量和可靠性，可以生成信息丰富的响应、生成准确的内容，精确地生成多语言翻译。    
![image](https://github.com/hellogxp/LLMs/assets/1506016/fc2be679-500c-472a-9cfa-12a1581f7ead)
Meta和UCL的一篇paper《Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks》中率先提出了RAG模型，涉及到了两个变种，RAG-Sequence和RAG-Token。论文中构建了一个参数记忆（seq2seq）和非参数记忆（retriever）相结合的端到端的混合概率模型，主要分为两个模块，Retriever=Query Encoder + Document Index和Generator=seq2seq model     
![image](https://github.com/hellogxp/LLMs/assets/1506016/12ffbc0e-d124-44a3-8088-171c4bb8185c)
分享一下Microsoft在2023分享的检索增强 LLM 方向的介绍    
![image](https://github.com/hellogxp/LLMs/assets/1506016/6170c86f-5abb-455f-910f-321933c5c1ae)
传统的搜索引擎如Google/Bing只具备检索能力，而新的语言模型 (LLM)通过预训练过程将大量数据和知识嵌入到其庞大的模型参数中，具备记忆能力。从这个角度来看，检索增强的LLM可以被看作是位于传统信息检索和LLM之间的中间状态。通过一些信息检索技术，相关信息可以被加载到LLM的工作内存（即上下文窗口）中，这也是LLM在单次生成时所能接受的最大文本输入。

### RAG的优势
RAG在NLP的众多领域具有具有明显的优势
#### 1. 可以获取广泛的外部知识
对于LM来说，它们有着数量巨大的预训练数据，但是在信息社会的今天，这些数据量也不能覆盖所有知识，尤其是我们之前提到的Long-tail Knowledge，很多领域或者专业知识，因为存储介质或者隐私等问题，大模型根本无法摄入，但是使用RAG就不同了，检索增强组件可以通过检索将这些外部数据源源源不断的输入到大语言模型中，使得以上问题迎刃而解。
#### 2. 确保模型能够访问最新、可靠的事实
大模型的预训练数据往往都不是最新的，比如ChatGPT的pre-training数据是几年前的，因此如果你询问其昨天刚发生的事情，它往往不能回答或者乱讲一通，也就是所谓的幻觉。但是RAG可以给LLMs输入最新的知识，给与它足够的上下文，来补足其短板，确保生成的回复始终基于最新和最相关的信息。
#### 3. 生成个性化且可验证的响应
RAG分为两阶段，检索和生成，在生成阶段，generator会将检索到的信息来源一并返回，用户可以访问模型的数据来源，以此来确定信息是否可靠。
#### 4. 保证数据安全
在RAG的phase1也就是检索阶段，retriever会检索与问题相关信息，如果在企业内部域，这些数据会存储在企业内部的数据库中，而不是作为预训练数据输入到大模型，极大减少了数据泄露的风险。如果很多企业内部或者涉及个人隐私的内容输入到LLMs中，如果遭到攻击比如Prompt Injection attack，很有可能造成数据泄露。同时有很多数据隐私保护协议也不允许将数据随意输入到大模型，比如GDPR、CCPA、HIPAA。    
![image](https://github.com/hellogxp/LLMs/assets/1506016/af5bc661-fce4-4670-89ba-654786567698)    
当然Prompt Injection只是其中一种攻击方法，Private and Data Safety of LLMs 又是一个很大的领域了，在此不表，总之，不是任何数据都适合灌入LLMs中。    
![image](https://github.com/hellogxp/LLMs/assets/1506016/dd9609c8-918f-4191-a104-f2dccba2c6f0)
#### 5. 增强相关性
在检索阶段会搜索与用户相关的信息结合原始prompt输入给LLMs，因为原始prompt可能会缺失关键信息，检索组件通过给LLMs输入上下文，丰富代言模型in-context learning的能力，相比于原始的prompt，产生的内容相关性会更强。

### 构建RAG应用
在RAG构建的过程中，我们会以LlamaIndex或者LongChain为例讲解，会涉及到其部分代码和概念。

#### 选择LLMs
在构建基于RAG的LLMs之前，我们首选需要选择一个大语言模型，不管是闭源的ChatGPT还是开源的Llama，当然可以使用多个。LLM可以用于多个阶段。
* 索引期间
在索引期间，可以使用 LLM 来确定数据的相关性（是否对其进行索引），或者可以使用 LLM 来汇总原始数据并为摘要建立索引。
* 查询期间
在查询期间，LLM 可以通过两种方式使用：
  * 在检索（从索引中获取数据）期间，LLMs可以获得一系列选项（例如多个不同的索引），并决定在哪里检索信息。代理LLM在这个阶段还可以使用工具来查询不同的数据源。
  * 在响应合成（将检索到的数据转换为答案）期间，LLM 可以将多个子查询的答案组合成一个连贯的答案，或者可以转换数据，例如从非结构化文本转换为 JSON 或其他输出格式。

例如在以下代码中，已实例化 OpenAI 并将其自定义为使用 gpt-4。
```python
from llama_index.llms import OpenAI
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
llm = OpenAI(temperature=0.1, model="gpt-4")
service_context = ServiceContext.from_defaults(llm=llm)
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(
    documents, service_context=service_context
)
```

#### Data Processing
##### Loading
数据加载，可以从多种数据源提取数据，无论是非结构化文本、PDF、结构化数据库还是其他应用程序的 API，以LlamaIndex为例，他有上百种connectors来获取你的数据，使用过flink的同学对connectors并不陌生，相当于一个连接器，连通数据源后，获取数据并将数据格式化为Document对象。Document是数据和有关该数据的元数据的集合。比如SimpleDirectoryReader，LlamaIndex 内置，可以读取多种格式，包括 Markdown、PDF、Word 文档、PowerPoint 幻灯片、图像、音频和视频。
```python
from llama_index import SimpleDirectoryReader

documents = SimpleDirectoryReader("./data").load_data()
```
当然 LlamaHub 有更多的 Reader，比如DatabaseReader，这个连接器可以直接连接SQL读取数据，然后将每一行转换为Document。
```python
from llama_index import download_loader

DatabaseReader = download_loader("DatabaseReader")

reader = DatabaseReader(
    scheme=os.getenv("DB_SCHEME"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASS"),
    dbname=os.getenv("DB_NAME"),
)

query = "SELECT * FROM users"
documents = reader.load_data(query=query)
```
##### Chunking
![image](https://github.com/hellogxp/LLMs/assets/1506016/0d48aac5-67b1-4dde-b39b-33924fa6c43e)    
在大语言模型 (LLM) 相关应用程序中处理大量文本数据时，分块起着至关重要的作用。这个过程将大段文本分解成更小的、易于理解的片段。这样，LLMs可以更有效地嵌入内容，并准确地从矢量数据库中检索相关数据。
* Chunk重要性
文档分块，这个过程非常重要，因为文档的分块策略在一定程度上决定了检索结果的质量，比如分块太大会导致噪声太大，夹杂太多无用信息，且可能超过LLMs的上下文窗口，目前GPT-4 Turbo的上下文窗口为128K。当然，太小也有问题，会丢失上下文信息。简单总结，为什么分块对于LLMs应用很重要：
  1. 保持语义相关性，也就是上下文相关性，这直接关乎回复准确率。
  2. 保持在大模型的上下文限制内，分块保证输入到LLMs的文本不会超过其token limitations。

* 短内容与长内容嵌入
我们需要将内容嵌入到向量中，关于向量嵌入的知识，可以查看Q&A章节中向量数据库的解释。内容的长短会影响到LLMs的输出。
短内容
比如句子级嵌入，向量专注于该句子的特定含义，可能会丢失段落或文档中更广泛的上下文信息。
长内容
嵌入整个段落或文档时候。会考虑整体上下文以及文本中句子和短语之间的关系。这可以产生更全面的矢量表示，捕获文本的更广泛的含义和主题。但是，较大的输入文本可能会引入噪音，可能削弱单个句子或短语的重要性，从而使在查询索引时找到精确匹配变得困难。

* Chunk的关注点
在分块前，你需要关注一下几个问题，才能制定好合理的分块策略。
    1. 内容类型，比如你的内容是文章，博客，大部头小说等长篇内容，还是推文，微博，朋友圈消息等短小的内容。
    2. 选择何种文本嵌入模型，不同的模型可能会适合不同的块尺寸。比如sentence-transformer在单个句子上有较好的效果，但是text-embedding-ada-002在256 or 512 tokens上表现更佳。
    3. 用户的查询输入长度和复杂度如何，是长还是短，是复杂还是简单，这也会影响我们的分块方式，因为查询嵌入也会使用相同的文本嵌入模型。
    4. 查询结果的用途，适用于智能QA，语义搜索，文本摘要？还是作为中间结果IR输入给LLMs，如果要送入LLMs，就需要考虑token limitation了。
这些问题考虑清楚了，就能找到一种性能和效率之间的trade-off。

* Chunk方法
分块的方法有多种，每种方法有各自的优缺点和适用场景。具体采用哪种方法需要结合自身场景来考虑。
  * 固定大小分块
这种方式简单粗暴，我们只需要定义好chunk中token的数量，chunk之间是否有重叠。正常来讲，我们会设置一定的重叠，这样块之间的上下文不会丢失。这种分块方法的好处显而易见，成本低，好操作，用不到NLP的库。
使用LLAMAindex库按句子对原始文本进行分块的示例代码如下：
```python
from llama_index import LLAMAIndex
import nltk
# 初始化LLAMAIndex
index = LLAMAIndex()
# 定义原始文本
raw_text = "This is the first sentence. This is the second sentence. This is the third sentence."
# 使用NLTK库进行句子划分
sentences = nltk.sent_tokenize(raw_text)
# 分块数据
chunks = index.chunk_data(sentences)
# 打印分块结果
for chunk in chunks:
    print(chunk)
```
LangChain分块的示例
```python
text = "..." # your text
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(
    separator = "\n\n",
    chunk_size = 256,
    chunk_overlap  = 20
)
docs = text_splitter.create_documents([text])
```
  * 内容感知分块
这个主要是根据内容的性质，应用一些比较复杂或者高级的方法。比如
Sentence splitting
很多模型都对句子级别嵌入进行了优化，我们有很多工具可以实现此功能：
NLTK：NLTK是一个流行的 Python 库，用于处理人类语言数据。它提供了一个句子标记器，可以将文本分割成句子，帮助创建更有意义的块。例如， NLTK 与 LangChain 结合使用：
```python
text = "..." # your text
from langchain.text_splitter import NLTKTextSplitter
text_splitter = NLTKTextSplitter()
docs = text_splitter.split_text(text)
```
spaCy：spaCy 也是一个用于 NLP 任务的强大 Python 库。它提供了复杂的句子分割功能，可以有效地将文本分割成单独的句子，从而在生成的块中更好地保​​留上下文。例如， spaCy 与 LangChain 结合使用：
```python
text = "..." # your text
from langchain.text_splitter import SpacyTextSplitter
text_splitter = SpaCyTextSplitter()
docs = text_splitter.split_text(text)
```
  * 递归分块
使用一组分隔符以分层和迭代的方式将输入文本划分为更小的块。如果初始尝试没有生成所需大小或结构的块，则该方法会使用不同的分隔符或标准在生成的块上递归调用，直到达到所需的块大小或结构。
给出一个在 LangChain 中使用递归分块的示例：
```python
text = "..." # your text
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 256,
    chunk_overlap  = 20
)
docs = text_splitter.create_documents([text])
```
  * 特例分析
我们这里分析两种特别的格式 Markdown 和 LaTeX 。
Markdown是一种轻量级标记语言，通常用于格式化文本，对于程序员来说并不陌生。通过识别 Markdown 语法（例如标题、列表和代码块），可以根据内容的结构和层次结构智能地划分内容，从而产生语义上更连贯的块。例如：
```python
from langchain.text_splitter import MarkdownTextSplitter
markdown_text = "..."
markdown_splitter = MarkdownTextSplitter(chunk_size=100, chunk_overlap=0)
docs = markdown_splitter.create_documents([markdown_text])
```
LaTex：写过paper的同学对这个再熟悉不过了，LaTeX 是一种基于TeX的排版系统，用于创建高质量的科技文档，如学术论文、报告、书籍和演示文稿。解析 LaTeX 也有专门的规律，例如：
```python
from langchain.text_splitter import LatexTextSplitter
latex_text = "..."
latex_splitter = LatexTextSplitter(chunk_size=100, chunk_overlap=0)
docs = latex_splitter.create_documents([latex_text])
```

当然，因为场景的多样性，以上方法仅供参考，如果没有现成的方法，可以根据以下的tips这来确定最佳的分块大小。
  * 数据预处理：在确定最佳块大小之前，首选要预处理数据以确保质量。例如，数据是从网络检索的，可能需要删除 HTML 标签或者其它无用元素来降噪。
  * 确定块大小范围：数据经过预处理后，下一步是确定潜在块大小范围。如前所述，选择分块方法需要考虑内容的性质（例如，短消息还是长的文档）、使用的嵌入模型及其功能（例如，令牌限制）。目标是在保留上下文和保持准确性之间寻找trade-off。这里就可以针对块大小进行多种尝试，包括用于捕获更细粒度语义信息的较小块（例如，128 或 256 个token）和用于保留更多上下文的较大块（例如，512 或 1024 个token）。
  * 评估每个块大小的性能：可以建立多个索引，来测试不同块大小的性能。创建不同大小的块，然后嵌入并保存在索引中。最后，运行一系列查询，来评估其质量，并比较不同块大小的性能。这是个迭代过程，可以针对不同的查询测试不同的块大小，直到可以确定内容和预期查询的最佳性能块大小。

总结
大多数情况下，对内容进行分块非常简单，但随着场景和分块内容的复杂，会遇到很多挑战。没有一种万能的分块解决方案，具体的分块策略还是要结合实际的应用场景和内容本身来确定。

##### Embedding
文本嵌入是文本的向量表示。由于机器需要数字输入来执行计算，因此文本嵌入是许多下游 NLP 应用程序的关键组成部分。例如，谷歌使用文本嵌入来支持他们的搜索引擎。文本嵌入还可以用于通过聚类在大量文本中查找模式，或者作为文本分类模型的输入，然而，文本嵌入的质量很大程度上取决于所使用的嵌入模型。 我们可以从MTEB（Massive Text Embedding Benchmar） 找到适合各种任务的最佳嵌入模型。
对于检索增强LLMs应用，我们会利用文本嵌入模型 ( Text Embedding Model ) 将文本块映射成一个固定长度的向量，存储在向量数据库中。检索时，对用户查询文本采用同样的文本嵌入模型映射成向量，然后基于向量相似度计算获取最相似的一个或者多个文本块。计算嵌入之间的相似度时，有很多方法可以使用（点积、余弦相似度等）。默认情况下，我们使用余弦相似度。    
![image](https://github.com/hellogxp/LLMs/assets/1506016/93f9e052-9065-4a69-81dd-3389862f3ff6)

#### Query Transformations & Rewriting
查询转换&改写，这个步骤不是必须的，但是建议采用，因为对于用户的查询，很多时候不建议直接使用，最好进行一些转换或者改写，来提高query的质量。
  * HyDE (Hypothetical Document Embeddings)：    
![image](https://github.com/hellogxp/LLMs/assets/1506016/ad92cdd9-9bc0-48d2-a348-cf6acaa3c924)    
假设性文档嵌入，是在paper《Precise Zero-Shot Dense Retrieval without Relevance Labels》中提出，给定初始查询，首先利用 LLM 生成一个假设的文档或者回复，然后以这个假设的文档或者回复作为新的查询进行检索，而不是直接使用初始查询，可以使用不同的技术和模型来实现。常见的方法包括：
    * 主题模型：将文档表示为潜在的主题分布，例如使用Latent Dirichlet Allocation（LDA）模型。
    * 上下文嵌入：考虑文档在上下文环境中的语义关系，例如使用Word2Vec或GloVe等词嵌入模型。
    * 基于注意力机制的模型：利用注意力机制来捕捉文档中重要的语义信息，例如Transformer模型。
给出一个LlamaIndex使用HyDE的例子：
```python
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader
from llama_index.indices.query.query_transform.base import HyDEQueryTransform
# load documents, build index
documents = SimpleDirectoryReader('../paul_graham_essay/data').load_data()
index = GPTSimpleVectorIndex(documents)
# run query with HyDE query transform
query_str = "what did paul graham do after going to RISD"
hyde = HyDEQueryTransform(include_original=True)
response = index.query(query_str, query_transform=hyde)
print(response)
```
这种转换在没有上下文的情况下可能会生成一个误导性的假设文档或者回复，从而可能得到一个和原始查询不相关的错误回复。因此，选择合适的假设和建模方法对于获得高质量的假设性文档嵌入至关重要，需要根据具体任务和数据的特点进行调整和优化。
  * 单步查询分解
Single-Step Query Decomposition，最近的一些研究（e.g. self-ask, ReAct）表明，当LLMs将问题分解为较小的步骤时，对于复杂问题的回答方面表现更好。对于知识增强的查询也是如此。
如果查询很复杂，知识库的不同部分会负责回答不同的subqueries。单步查询分解功能通过数据收集将复杂的问题转换为简单的问题，以帮助提供原始问题的子答案。这对于组合图尤其有用。在组合图中，查询可以路由到多个子索引，每个子索引代表整个知识库的一个子集。查询分解允许我们将查询转换为针对任何给定索引的更合适的问题。    
![image](https://github.com/hellogxp/LLMs/assets/1506016/480837a0-6dd1-4779-ba50-a828baf62413)    
相应的示例代码片段：
```python
# Setting: a list index composed over multiple vector indices
# llm_predictor_chatgpt corresponds to the ChatGPT LLM interface
from llama_index.indices.query.query_transform.base import DecomposeQueryTransform
decompose_transform = DecomposeQueryTransform(
    llm_predictor_chatgpt, verbose=True
)
# initialize indexes and graph
# set query config
query_configs = [
    {
        "index_struct_type": "simple_dict",
        "query_mode": "default",
        "query_kwargs": {
            "similarity_top_k": 1
        },
        # NOTE: set query transform for subindices
        "query_transform": decompose_transform
    },
    {
        "index_struct_type": "keyword_table",
        "query_mode": "simple",
        "query_kwargs": {
            "response_mode": "tree_summarize",
            "verbose": True
        },
    },
]
query_str = (
    "Compare and contrast the airports in Seattle, Houston, and Toronto. "
)
response_chatgpt = graph.query(
    query_str, 
    query_configs=query_configs, 
    llm_predictor=llm_predictor_chatgpt
)
```

  * 多步查询转换
通过逐步迭代，将一个复杂的查询逐渐分解为更小、更简单的子查询。在多步分解中，每一步只完成一个特定的分解或处理步骤，并将分解结果作为下一步的输入。这种分解策略可以更加深入地理解查询，提供更细致的分解和处理，但可能需要更多的计算和时间成本。    
![image](https://github.com/hellogxp/LLMs/assets/1506016/0887fc11-359a-4d2a-8dbc-5329f6b175f6)    
在LlamaIndex中有个 QueryCombiner 类，可以实现此功能。
```python
from llama_index.indices.query.query_transform.base import StepDecomposeQueryTransform
# gpt-4
step_decompose_transform = StepDecomposeQueryTransform(
    llm_predictor, verbose=True
)
response = index.query(
    "Who was in the first batch of the accelerator program the author started?",
    query_transform=step_decompose_transform,
)
print(str(response))
```
关于Rewriting，最近有一篇较新的paper《Query Rewriting for Retrieval-Augmented Large Language Models》可以参考，这篇paper专门从查询重写的角度引入了一个新的框架，有点意思，此处不细说，有兴趣可细看。    
![image](https://github.com/hellogxp/LLMs/assets/1506016/e81705de-59e2-48a7-bd67-f347b153525b)

#### Retrieving
##### Retrieval
检索过程，从 Index 中找到并返回与查询最相关的文档。最常见的检索类型是“top-k”语义检索，当然许多其他检索策略，在此不细说。
##### Postprocessing
后处理，检索到的 Node 可以选择性地rerank、转换或过滤等，常用策略包括：
* 相关性过滤：根据文档与查询的相关性得分进行过滤筛选，只保留得分较高的文档。可以使用基于向量空间模型（Vector Space Model）、BM25等算法计算文档与查询的相关性得分。
* 重要性排序：根据文档的重要性指标对文档进行排序。常用的指标有PageRank、TF-IDF等，可以根据这些指标对文档进行降序排序。
* 时间排序：根据文档的时间戳或发布日期对文档进行排序，以便最新的文档排在前面。这在新闻、论坛帖子等领域中常见。
* 个性化排序：根据用户的偏好、历史行为等个性化信息，对文档进行排序。可以使用协同过滤、推荐系统等技术来根据用户的兴趣和行为进行个性化排序。
* 多维度排序：结合多个指标或特征，如相关性得分、重要性、时间等，进行综合排序。可以使用加权求和、排序算法等方法来综合考虑多个排序因素。对排序结果还可以进一步处理和调整，以修正潜在的问题。例如，去除重复文档、处理异常值、平滑排序得分等。

#### Response synthesis
将查询、最相关的数据和提示组合起来并发送给LLMs以生成最终的回复。    
![image](https://github.com/hellogxp/LLMs/assets/1506016/97823de9-a124-4374-9988-7ad281829c48)    
响应合成也有很多模式和策略，这里以LlamaIndex为例介绍几种Response Mode
* refine
通过顺序浏览检索到的每个文本块来创建和完善答案，这种模式下，每个检索到的块都要进行单独的 LLM 调用。具体流程为：前一个查询的答案和下一个块（以及原始问题）将在下一个查询中使用。依此类推，直到解析完所有块。如果使用的是ChatGPT等闭源LLMs，就要考虑成本了。
* compact
与 refine 类似，但预先压缩Chunk，从而减少 LLM 调用。将尽可能多的块填充在一个窗口上下文中，如果文本太长而无法容纳在一个提示中，则需要将其拆分，同时允许文本块之间存在一些重叠以不丢失上下文。
* simple_summarize
截断所有文本块以适合单个 LLM 提示。适合快速总结，但可能会因截断而丢失细节。
当然还有很多其他的方法，可以根据自己的需求去探索。

#### Evaluating

### RAG技术模型和架构
#### 技术模型
* RAG-Sequence
在此模型中，使用相同的检索文档来预测目标序列中的每个标记。它通过在整个生成过程中依赖单个文档来保持一致性。
* RAG-Token
可以根据不同的文档来预测目标序列中的不同标记。这允许更大的灵活性，因为每个token都可以从最相关的上下文中受益。

#### 开发平台
* LlamaIndex
LlamaIndex是一个服务于 LLM 应用的数据框架，提供外部数据源的导入、结构化、索引、查询等功能。    
![image](https://github.com/hellogxp/LLMs/assets/1506016/e4d88a7b-ed10-49d6-b1e5-1668077c4ec1)
* LangChain
LangChain也是一种流行的LLM应用开发框架。LangChain中也包含了一些用于增强LLM的相关组件。然而，相较而言，LlamaIndex更专注于增强LLM这个相对较小的领域，而LangChain则覆盖了更广泛的领域。例如，LangChain包括LLM的链式应用、代理人的创建和管理等功能。下面的图示展示了LangChain中检索（Retrieval）模块的整体流程，包括数据加载、变换、嵌入、向量存储和检索。整体处理流程与LlamaIndex是相同的。    

![image](https://github.com/hellogxp/LLMs/assets/1506016/2e9ab60a-1dba-484f-95a7-7efd9467a64d)    

在实际应用中，langchain和llamaindex已经成功实现了RAG架构，并通过不断迭代和丰富，发展成为一个功能完备的编排框架，能够支持各种LLM应用的常见形态，如聊天机器人、协作助手和代理人等。从架构图可以看出，除了LLM和编排服务，向量数据库在整个系统中也起着重要的作用。它不仅存储了从领域知识库中导入并进行嵌入的知识，还保存了用户与大模型对话的信息，被称为"memory"，向量数据库的描述详见Q&A章节。

### User cases
基于RAG的LLM系统可以构建众多应用，包括问答系统，聊天机器人和虚拟助理，内容摘要，内容生成，跨语言应用，知识库和专家系统等。
#### 问答式搜索服务
目前开源的方案有Havenask-LLM的解决方案，目前Havenask支持问答的方式就是前文所述经典方案“文档切片+向量检索+大模型生成答案”的方案。    

![image](https://github.com/hellogxp/LLMs/assets/1506016/683b32db-943d-408a-86a0-8f1f65e2be7c)    

Havenask-LLM的架构如下图所示，整体分为3个部分：    

![image](https://github.com/hellogxp/LLMs/assets/1506016/e2c385d0-d851-4aee-94cf-1eae0f7821b9)    

落地场景：    

![image](https://github.com/hellogxp/LLMs/assets/1506016/98a695c3-27f4-48b7-aa96-9852863cee07)    

#### ChatGPT retrieval plugins
ChatGPT 检索插件是一个使用检索增强LLM的典型案例。用户与 ChatGPT 的对话及其嵌入表示可以被存储以供以后检索。
下图描述了检索插件、数据库和 OpenAI/ChatGPT 服务之间的交互，可以实现个人或组织文档的语义搜索和检索。它允许用户通过用自然语言提出问题或表达需求，从其数据源（例如文件、笔记或电子邮件）获取最相关的文档片段。企业可以使用此插件通过 ChatGPT 向员工提供内部文档。    

![image](https://github.com/hellogxp/LLMs/assets/1506016/ab7c21b5-a126-4baf-a709-6b133194038e)    
### RAG 面临的挑战和方案
检索增强生成（Retrieval-Augmented Generation，RAG）方法通过检索相关知识来减少LLMs在事实错误方面的问题，降低了LLMs在中长尾任务中的事实错误率。然而，这种方法可能面临以下问题：

1. 这些方法不加区别地检索和合并了一定数量的检索文段，无论是否需要检索，以及文段是否相关。这种做法可能降低了LLMs的多功能性，或导致生成质量下降，它们不加区别地检索文段，无论事实支持是否有帮助。

2. 生成结果未必与检索的相关文段一致，因为这些模型没有明确训练以利用和遵循所提供文段的事实。    

![image](https://github.com/hellogxp/LLMs/assets/1506016/4ff0a387-e2a7-4708-80c9-eec552245793)    

论文《Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection
》引入了一种名为自我反思检索增强生成（SELF-RAG）的新框架，通过按需检索和自我反思来提高语言模型（LLM）的生成质量，特别是事实准确性，同时不损害其多功能性。该研究通过以端到端的方式训练LLMs，使其能够反思自身的生成过程，通过生成任务的输出和特殊间断标记（即反思token）来实现。这些反思token分为检索token和评论token，分别表示检索需求和生成质量（图右侧）。具体的方法如下：

1. 给定一个输入提示和previous的生成，SELF-RAG首先确定，在继续生成中增加检索文段是否有帮助。如果yes，它会输出一个检索标记，以便按需调用一个检索模型。
2. SELF-RAG同时处理多个检索文段，评估它们的相关性，然后生成相应的任务输出。
3. 生成评论标记来反思自己的输出并选择在事实准确性和整体质量方面最好的生成。这个过程与传统的RAG（图左侧）不同，后者不管检索是否有必要（例如，底部示例不需要事实知识），都会一律检索固定数量的文档进行生成，并且不会关注生成质量。

另外，SELF-RAG针对每个部分都提供引文，并附带自我评估，以确定生成的输出是否受到相关文段的支持，从而简化了事实验证的过程。SELF-RAG通过将反思标记整合为来自扩展模型词汇表的下一个标记预测，训练任意的语言模型（LM）来生成文本。受到强化学习中奖励模型的启发，论文通过在原始语料库中插入预训练的评论模型生成的反思token，从而避免了托管评论模型的训练过程，降低了开销。评论模型在输入、输出和相应的反思标记数据集上进行了监督学习，这些数据集是通过提示专有的LLM（如GPT-4）收集的。尽管从使用控制标记来启动和指导文本生成的研究中汲取灵感，论文中训练的LLM在生成输出的每个部分后使用评论标记来评估自身的预测，作为生成输出的一部分。

SELF-RAG进一步提供了可定制的解码算法，以满足硬性或软性约束，这些约束由反思token的预测定义。论文提到的推理算法能够：

1. 在不同的下游应用程序中灵活调整检索频率；
2. 通过使用反思标记，可以根据基于段的权重线性和算法，定制模型的行为，以满足用户的偏好。

实验证据表明，SELF-RAG在六个任务上明显优于具有更多参数的经过预训练的LLMs，以及广泛采用的具有更高引用准确性的RAG方法。特别是，在四个任务中，SELF-RAG的性能优于具有检索增强功能的ChatGPT、Llama2-chat和Alpaca。分析证明，使用反思标记进行训练和推理对整体性能提升至关重要，并且能够有效地进行模型自定义，例如在引文预测和完整性之间进行权衡。

3. 数据安全
因为我们与retrieve的过程，这个阶段实际上也可能被攻击，比如 Indirect Prompt Injection。    
![image](https://github.com/hellogxp/LLMs/assets/1506016/92b8e6c3-84fd-4f0c-a79c-077438a6fc2a)    
之前讲过，LLMs的数据安全又是一个很大的领域了，至于怎么来避免这些攻击，本篇不表了。    
![image](https://github.com/hellogxp/LLMs/assets/1506016/6fe7c44b-539b-4544-a45a-7cb407430e3c)    

### Conclusion
本文从大模型的属性出发，探讨了大模型架构RAG的产生背景和基本模式，看到了大模型+搜索为LLMs落地企业带来的可能性，以及产生的创新场景，同时讨论了RAG改进方向和方案，然而关于LLMs的一切才刚刚开始，在实际生产过程中，有大量的问题尚待解决，比如效果，内容安全与隐私保护，内容可信，性能等多个方面，期待下次分享。
### Q&A
#### Spearman
$$\rho = 1 - \frac{6\sum d_i^2}{n(n^2 - 1)}$$
Spearman是一个常见的统计学概念，用于衡量两个变量之间的相关性。它基于等级或顺序数据，而不是数值数据。
Spearman相关性系数（Spearman's rank correlation coefficient）通过将每个变量的值转换为等级或顺序，然后计算这些等级或顺序之间的相关性来衡量变量之间的相关性。它是一种非参数测试方法，适用于无法满足正态分布假设的数据。
Spearman系数的取值范围在-1到1之间，其中-1表示完全逆向的相关性，0表示没有相关性，1表示完全正向的相关性。Spearman系数的计算方法是通过计算等级或顺序之间的差异平方和的比率来得出的。Spearman系数可用于研究两个变量之间的关系，无论是线性关系还是非线性关系。它通常用于社会科学研究中，例如心理学和教育研究，以及市场研究和调查等领域。
需要注意的是，Spearman系数只能衡量两个变量之间的单一关系，不能用于多个变量之间的相关性分析。
#### 文本嵌入模型
文本嵌入模型是一种用于将文本数据映射到低维向量空间的机器学习模型。它的目标是将文本的语义信息编码为向量表示，使得具有相似语义的文本在向量空间中距离较近，而语义上不相关的文本在向量空间中距离较远。

常见的文本嵌入模型包括Word2Vec、GloVe、FastText和BERT等。这些模型使用不同的方法和算法来捕获文本的语义信息。Word2Vec和GloVe是基于词级别的模型，通过分析词的上下文关系来生成词向量，目前很少使用。而BERT则是一种基于深度神经网络的预训练模型，它通过双向语言模型预训练和下游任务微调的方式，能够更好地理解文本语义和上下文信息。在MTEB Leaderboard榜单可以看到文本嵌入模型的排行榜，目前排在第一位的是COHERE模型。    

![image](https://github.com/hellogxp/LLMs/assets/1506016/597b4e7a-0dbd-4417-af6a-c5602b93028a)    

#### 相似向量检索
相似向量检索是一种用于在向量空间中查找最相似向量的技术。在相似向量检索中，通过计算向量之间的相似度或距离度量来确定向量之间的相似程度，从而找到与给定查询向量最相似的向量。
相似向量检索常用于处理大规模的向量数据集，例如文本、图像或音频数据。它可以用于各种应用场景，如信息检索、推荐系统、图像搜索和嵌入式系统等。
在相似向量检索中，常见的相似度或距离度量包括余弦相似度、欧氏距离、曼哈顿距离和汉明距离等。通常情况下使用余弦相似度可满足大部场景。
#### 向量数据库
为了确保大模型生成高质量的输出，RAG架构的核心在于检索，而检索的核心是相关性。关键在于如何提高检索相关性。在这方面，向量检索相对于传统的文本相关性检索具有显著的优势，即语义相关性。    

![image](https://github.com/hellogxp/LLMs/assets/1506016/4654200b-93c9-4bdf-b03c-bdaa8bf6bfb1)    

以图中的检索为例，假设资料库中存在与鱼搭配的葡萄酒。在进行搜索时，如果搜索词为"有海鲜搭配的酒"，传统搜索方法无法检索到相关结果，因为"fish"和"seafood"在文本上并没有直接的相关性。然而，对于使用嵌入向量的方法来说，海鲜和鱼之间存在语义相关性，因此可以被检索到。而向量数据库恰好用于存储嵌入向量，并进行向量相似性的计算。在LLM这个场景下，使用向量数据库是最合适的选择。

1. 对于大模型而言，它们的本质都是进行嵌入向量的计算。它们采用相同的嵌入方式，可以相互协同工作，发挥更好的效果。
2. 向量数据库在大规模数据检索上具有快速的速度，可以弥补大模型推理速度较慢的问题。向量数据库采用了多种相似性计算算法，例如ANN（近似最近邻）、LSH（局部敏感哈希）、PQ（乘积量化）等，进一步提高了检索效率。
3. 向量数据库天生具备多模态的能力。实际上，无论是文本、图片、声音还是视频，都可以通过向量表示进行存储和检索。这样就能够统一地存储不同模态的数据，对于大模型而言，也具有统一的趋势。从技术发展的角度来看，使用向量表示进行匹配具有较高的准确度。

对于高效的相似性搜索和向量聚类，Facebook团队开源的Faiss是一个非常好的选择。Faiss是一个功能强大的库，它实现了在任意大小的向量集合中进行搜索的多种算法。除了可以在CPU上运行，一些算法还支持GPU加速。Faiss包括了多种相似性检索算法，具体使用哪种算法则需要综合考虑数据量、检索频率、准确性和检索速度等因素。

### Reference
[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
](https://arxiv.org/abs/2005.11401)    
[Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/pdf/2212.10496.pdf)    
[Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection
](https://arxiv.org/abs/2310.11511)    
[Large Language Models Struggle to Learn Long-Tail Knowledge
](https://arxiv.org/abs/2211.08411)    
[Extracting Training Data from Large Language Models
](https://arxiv.org/abs/2012.07805)    
[Retrieval-based Language Models and Applications
](https://acl2023-retrieval-lm.github.io/)    
[Emerging Architectures for LLM Applications](https://a16z.com/emerging-architectures-for-llm-applications/)    
[RAG vs Finetuning — Which Is the Best Tool to Boost Your LLM Application](https://towardsdatascience.com/rag-vs-finetuning-which-is-the-best-tool-to-boost-your-llm-application-94654b1eaba7)    
[Chunking Strategies for LLM Applications](https://www.pinecone.io/learn/chunking-strategies/)    
[OpenAI/ChatGPT retrieval plugin and PostgreSQL on Azure
](https://techcommunity.microsoft.com/t5/azure-database-for-postgresql/openai-chatgpt-retrieval-plugin-and-postgresql-on-azure/ba-p/3826411)
[mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard)    
[Generative-Vs-Discriminative-Models](https://krutikabapat.github.io/Generative-Vs-Discriminative-Models/)    
[About Generative and Discriminative models](https://medium.com/@jordi299/about-generative-and-discriminative-models-d8958b67ad32)    
[LangChain](https://python.langchain.com/docs/get_started/introduction)    
[LlamaIndex](https://www.llamaindex.ai/)    
[MTEB: Massive Text Embedding Benchmark](https://huggingface.co/blog/mteb)    
[Query Rewriting for Retrieval-Augmented Large Language Models](https://arxiv.org/abs/2305.14283)    
[A Gentle Introduction to Retrieval Augmented Generation](https://wandb.ai/cosmo3769/RAG/reports/A-Gentle-Introduction-to-Retrieval-Augmented-Generation-RAG---Vmlldzo1MjM4Mjk1#rag-components-)    
[Measuring and Narrowing the Compositionality Gap in Language Models](https://arxiv.org/abs/2210.03350)    
[ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)    
[Value-based distance between the information structures](https://arxiv.org/abs/1908.01008#:~:text=We%20dene%20the%20distance%20between,across%20all%20zero%2Dsum%20games.)    
[What is retrieval-augmented generation](https://research.ibm.com/blog/retrieval-augmented-generation-RAG)    
[ChatGPT Retrieval Plugin](https://github.com/openai/chatgpt-retrieval-plugin)    
[Efficient Information Retrieval with RAG Workflow](https://medium.com/international-school-of-ai-data-science/efficient-information-retrieval-with-rag-workflow-afdfc2619171)    
[Prompt Injection attack against LLM-integrated Applications](https://arxiv.org/pdf/2306.05499.pdf)    
[Mitigating Stored Prompt Injection Attacks Against LLM Applications](https://developer.nvidia.com/blog/mitigating-stored-prompt-injection-attacks-against-llm-applications/#:~:text=Prompt%20injection%20attacks%20are%20a,on%20and%20has%20access%20to.)    
[Not what you’ve signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/pdf/2302.12173.pdf)    





