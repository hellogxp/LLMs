## CDC简介

CDC是Change Data Capture（变更数据获取）的简称。核心思想是，监测并捕获数据库的变动（包括数据或数据表的插入、更新以及删除等），将这些变更按发生的顺序完整记录下来，写入到消息中间件中以供其他服务进行订阅及消费。

CDC允许从数据库实时捕获已提交的更改，并将这些更改传播到下游消费者。在需要保持多个异构数据存储同步（如 MySQL 和 ElasticSearch）的用例中，CDC 变得越来越流行，它解决了双写和分布式事务等传统技术存在的挑战。

在广义的概念上，只要是能捕获数据变更的技术，我们都可以称之为 CDC 。目前通常描述的 CDC 技术主要面向数据库的变更，是一种用于捕获数据库中数据变更的技术。CDC 技术的应用场景非常广泛：

- **数据同步**：用于备份，容灾；
- **数据分发**：一个数据源分发给多个下游系统；
- **数据采集**：面向数据仓库 / 数据湖的 ETL 数据集成，是非常重要的数据源。

CDC 的技术方案非常多，目前业界主流的实现机制可以分为两种，基于查询和基于Binlog两种方式，这两种之间的区别：

|  | 基于查询的CDC | 基于Binlog的CDC |
| --- | --- | --- |
| 开源产品 | Sqoop、Kafka JDBC Source | Canal、Maxwell、Debezium |
| 执行模式 | Batch | Streaming |
| 是否可以捕获所有数据变化 | 否 | 是 |
| 延迟性 | 高延迟 | 低延迟 |
| 是否增加数据库压力 | 是 | 否 |


-  **基于查询的 CDC** 
   - 离线调度查询作业，批处理。把一张表同步到其他系统，每次通过查询去获取表中最新的数据；
   - 无法保障数据一致性，查的过程中有可能数据已经发生了多次变更；
   - 不保障实时性，基于离线调度存在天然的延迟。
-  **基于日志的 CDC**
实时消费日志，流处理，例如 MySQL 的 binlog 日志完整记录了数据库中的变更，可以把 binlog 文件当作流的数据源；
保障数据一致性，因为 binlog 文件包含了所有历史变更明细；
保障实时性，因为类似 binlog 的日志文件是可以流式消费的，提供的是实时数据。
对比常见的开源 CDC 方案，我们可以发现：
![](https://ata2-img.oss-cn-zhangjiakou.aliyuncs.com/neweditor/53eb8b2b-e144-4fea-bbdf-17e1a1e17393.png#id=G9Uz9&originHeight=487&originWidth=866&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=) 
-  对比增量同步能力 
   - 基于日志的方式，可以很好的做到增量同步；
   - 而基于查询的方式是很难做到增量同步的。
-  对比全量同步能力，基于查询或者日志的 CDC 方案基本都支持，除了 Canal。 
-  而对比全量 + 增量同步的能力，只有 Flink CDC、Debezium、Oracle Goldengate 支持较好。 
-  从架构角度去看，这里将架构分为单机和分布式，这里的分布式架构不单纯体现在数据读取能力的水平扩展上，更重要的是在大数据场景下分布式系统接入能力。例如 Flink CDC 的数据入湖或者入仓的时候，下游通常是分布式的系统，如 Hive、HDFS、Iceberg、Hudi 等，那么从对接入分布式系统能力上看，Flink CDC 的架构能够很好地接入此类系统。 
-  在数据转换 / 数据清洗能力上，当数据进入到 CDC 工具的时候是否能较方便的对数据做一些过滤或者清洗，甚至聚合 
   - 在 Flink CDC 上操作相当简单，可以通过 Flink SQL 去操作这些数据；
   - 但是像 DataX、Debezium 等则需要通过脚本或者模板去做，所以用户的使用门槛会比较高。
-  另外，在生态方面，这里指的是下游的一些数据库或者数据源的支持。Flink CDC 下游有丰富的 Connector，例如写入到 TiDB、MySQL、Pg、HBase、Kafka、ClickHouse 等常见的一些系统，也支持各种自定义 connector。 

## DBlog概述

在 MySQL 和 PostgreSQL 这样的数据库中，事务日志是 CDC 事件的来源。由于事务日志的保留期通常有限，所以不能保证包含全部的更改历史。因此，需要转储来捕获源的全部状态。目前的CDC 项目不能满足Netflix的需求的，例如，在转储完成之前暂停日志事件的处理，缺少按需触发转储的能力，或者使用表级锁来阻止写流量的实现。这是 DBLog 的开发动机，它在通用框架下提供日志和转储处理。

DBLog 的部分功能如下：

- 按顺序处理捕获到的日志事件。
- 随时进行dumps，跨所有表，针对一个特定的表或者针对一个表的具体主键。
- 以chunk的形式进行查询（转储），log与dump events交错。通过这种方式，日志处理可以与转储处理一起进行。如果进程终止，它可以在最后一个完成的块之后恢复，而不需要从头开始。这还允许在需要时对转储进行调整和暂停。
- 不会获取表级锁，这可以防止影响源数据库上的写流量。
- 支持任何类型的输出，因此，输出可以是流、数据存储甚或是 API。
- 设计充分考虑了高可用性。因此，下游的消费者可以放心，只要源端发生变化，它们就可以收到变化事件。

DBlog 可以在不中断CDC过程，不锁表的情况下，在任意时刻捕获表的full state。且能实现随时终止、随时恢复的能力，我们在建立一个OLTP数据库的表到大数据链路的过程，通常分为两步：

1. bootstrap：历史存量数据的初始“导入”；
2. incremental ingestion：增量摄取，也即我们常说的CDC的过程。

在老的方案中，都是拆分为这两步串行来做的。我们称之为：全量+增量以及全量增量衔接。这种方案简单粗暴，且两步必须是串行完成，如果bootstrap没做完，增量的流数据是没法先摄取到数据湖的。

DBLog 通过事务日志捕获变更事件，通过 select 来获取数据的全状态。根据主键将表切分为多个 chunk，每个 chunk 的 select 和事务日志同时进行，为了保证事件的历史顺序，DBLog 在源数据库中维护了一个单行单列的表作为辅助，通过在 select 前后分别更新该记录使得在事务日志中多了两个事件 lw(低水位) 和 hw(高水位)，然后将查询出来的事件和 [lw, hw] 之间的事件做以下算法，生成一个新事件流。

## DBlog的实现

首先看下DBLog的High Level架构：
![](https://ata2-img.oss-cn-zhangjiakou.aliyuncs.com/neweditor/64bdaab2-9f6d-45cc-b859-8c8cafc930d4.png#id=iY1QK&originHeight=656&originWidth=998&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)
架构比较简单，DBLog服务基于ZK做了个简单的HA，并利用ZK来存储处理过程中的State。从上游摄取change log，然后从表里查到特定的记录行并与其interleave（交织），最后再output。

### 两个核心的设计点

#### Chunk

将要捕获的表切分为一个个小的“块”。它会对表进行查询，然后对“数值型”的主键做个“升序”排序。在这个“即时查询得到的视图”上，才会将其切割为记录数相等的“块”。形如下图：
![](https://ata2-img.oss-cn-zhangjiakou.aliyuncs.com/neweditor/8c228693-6ecb-4511-8cc4-2c8a822812dc.png#id=tau9w&originHeight=642&originWidth=740&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)
不管删除机制是怎样，它本质上都可以看成是串行、有序的“记录表”。只要是对现存的记录，重排若干次其“顺序性”不会发生改变。chunk里的数据其实是即席查询的“快照视图”的切片。

#### Watermark

因为change log本质上是个“流”，流是unbounded，但业界通常都会选择安插一些特殊的“标记”来表示“流”向前演进的“进展”。而自Google在其流计算系统MillWheel中将这种标记称之为“Watermark”之后，它便在各种系统里出现，并在Flink流行后而被进一步发扬光大。

DBLog在设计中也引入了“Watermark”。它通过在源库中，新建一个称之为“Watermark”的表，然后通过向这个表里插入一个UUID来表示对watermark的更新（这里只是使用了watermark在流里进行切片的作用，并没有利用它来度量流的演进状态）。由于change log的scope在数据库实例级别，因此，对watermark表的操作所产生的"Watermark"的change log会自动interleave到change log stream中，且这些都是数据库自行完成的，毫无侵入性。

假设我们在某一时刻，生成了一个watermark=L，在未来的某一时刻，又生成了一个watermark=H。当然，在这之间会有很多事务性的insert/update所产生的正常change log。那么，L~H框出了一个bounded dataset，也即“window”，window里是正常DML操作所产生的change log。

### Watermark-based chunk selection

DBLog会基于watermark来进行chunk的选择，然后踢掉重复的记录，将chunk里的表记录interleave到change log流里。以下是这段算法的伪代码：
![](https://ata2-img.oss-cn-zhangjiakou.aliyuncs.com/neweditor/55004581-6e1e-4c6f-9942-b7e5cbdc988e.png#id=Rbbxl&originHeight=1330&originWidth=966&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)
拆解一下这段算法：

第一步到第四步的图解如下：
![](https://ata2-img.oss-cn-zhangjiakou.aliyuncs.com/neweditor/055dcd68-3f70-4e3e-8a9e-eb0a39297b86.png#id=feeOp&originHeight=554&originWidth=972&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)
首先，源数据库，一直在不断地产生change log，这一点是RDBMS的行为且不会也无需中断。DBLog程序会顺序读取并解析，这个是DBLog可以控制的。

流程开始时，DBLog会暂停处理（1），然后它会先生成一个低水位的“watermark” L（2）。接着，它会进行一个chunk的选择（3），这里补充几个点：

- chunk 是大小是固定的且可配的；
- chunk 的偏移状态保存在ZooKeeper里；
- chunk里的数据不用被预先存起来，而是可以每次通过ad-hoc拿到；
- chunk体现的是查询时的快照。

chunk里数据集拿到后，再生成一个高水位的“watermark” H（4）。拿到chunk数据集之后，再生成新的watermark，意味着在新的watermark生成之前，这个数据集对它是“可见”的。此时L~H也形成了一个change log窗口，也即另一个dataset。因此，现在我们就有了两个dataset。

那么，接下来要做什么：
![](https://ata2-img.oss-cn-zhangjiakou.aliyuncs.com/neweditor/90150658-bf18-4936-8364-b2f5f8d808b3.png#id=u6L1R&originHeight=534&originWidth=940&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

在chunk中，踢掉两个数据集中overlapping的数据，剩下的不重复的数据留在chunk中。我们来解释下，为什么要踢掉重叠的数据。在同一时间范围内（L~H），我们看到了chunk中存在的，未被处理的数据“记录”（静态状态）。又看到了change log窗口。而根据“流表二象性”，流通过回放事件，可以形成表某时刻的快照。所以，此时K1、K3就没必要保存在chunk里的，因为最终如果整个change log流被回放的话，会将K1、K3体现在表中。最后：我们将chunk里快照的剩余数据安插在窗口最后。

![](https://ata2-img.oss-cn-zhangjiakou.aliyuncs.com/neweditor/83d6be5e-ff6a-41ea-8a25-fc49b6320cd7.png#id=aNVYP&originHeight=418&originWidth=912&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

也即将表里的静态数据（状态）增补进去。看到的就是一个相对更完整的：change log流与static data state交织的完整流。当然再之后，如果K2~K6 仍然发生了改变，也不影响正确性。因为，插入的位置在H之前，而其他的变更必然在H之后，所以不会影响流表二象性的语义。

随着chunk被一轮接一轮地处理，更多的static data state被interleave到一个super change log里去。我们就可以在不中断事务日志或业务过程的情况下，渐进式地捕获所有的“静态数据”状态（也即我们之前所提到的bootstrap的目的）。最后，我们将这个super change log回放到下游的湖仓里，如Hudi表，那么就能完全反应上游表的全貌。

> 流表二象性
>  
> - streams -> tables：一个关于变更的数据流随着时间聚集产生了一个数据表；
> - tables->streams：对一个表的变更的观察，伴随着时间的变化产生了一个数据流（observation这一过程，体现了离散的概念）；
![](https://ata2-img.oss-cn-zhangjiakou.aliyuncs.com/neweditor/f4a3cf56-d43b-4020-b2e5-ab5a86e169df.png#id=ZOs7o&originHeight=632&originWidth=1414&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)


### 优缺点总结

论文解决的是一个实际的工程问题，逻辑并不复杂，它巧妙地应用了流计算里的一些设计，本质上还是“流表二象性”的一个应用。且这个设计已经著名的CDC框架debezium所借鉴并扩展实现。

- 优点
将获取表的full state的过程，从传统的串行化、“锁表”等代价很大的方式，替换成现在的并行、低成本的、change log和表记录“均衡”交织的方式，可以说是一大进步。论文里所有的这些解法，都没有涉及到市面上常见RDBMS独有的设计（如，yelp在其博客中介绍了一种依赖于Mysql Blackhole引擎的bootstrap设计6，其通用性上有些劣势），从而使得其使用场景收到局限。使用ZK保存了chunk处理的状态，可以暂停或恢复。
- 弊端
由于chunk的设计，依赖一些前提假设：数值型的自增主键，来确保可排序，使得其应用场景收到了一定的限制。

## Flink CDC项目背景（动机）

### Dynamic Table & ChangeLog Stream

Flink 有两个基础概念：Dynamic Table 和 Changelog Stream。
![](https://ata2-img.oss-cn-zhangjiakou.aliyuncs.com/neweditor/91277c22-c1bf-4824-b09b-2586a94dbb9d.png#id=NLj8J&originHeight=844&originWidth=1500&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

- Dynamic Table 就是 Flink SQL 定义的动态表，动态表和流的概念是对等的。参照上图，流可以转换成动态表，动态表也可以转换成流。
- 在 Flink SQL中，数据在从一个算子流向另外一个算子时都是以 Changelog Stream 的形式，任意时刻的 Changelog Stream 可以翻译为一个表，也可以翻译为一个流。

联想下 MySQL 中的表和 binlog 日志，会发现：MySQL 数据库的一张表所有的变更都记录在 binlog 日志中，如果一直对表进行更新，binlog 日志流也一直会追加，数据库中的表就相当于 binlog 日志流在某个时刻点物化的结果；日志流就是将表的变更数据持续捕获的结果。这说明 Flink SQL 的 Dynamic Table 是可以非常自然地表示一张不断变化的 MySQL 数据库表。
![](https://ata2-img.oss-cn-zhangjiakou.aliyuncs.com/neweditor/6f2332ad-21a7-4d93-a6bd-6a786abed25e.png#id=yqtOZ&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

Flink CDC选择了 Debezium 作为底层采集工具。Debezium 支持全量同步，也支持增量同步，也支持全量 + 增量的同步，非常灵活，同时基于日志的 CDC 技术使得提供 Exactly-Once 成为可能。

将 Flink SQL 的内部数据结构 RowData 和 Debezium 的数据结构进行对比，可以发现两者是非常相似的。

- 每条 RowData 都有一个元数据 RowKind，包括 4 种类型， 分别是插入 (INSERT)、更新前镜像 (UPDATE_BEFORE)、更新后镜像 (UPDATE_AFTER)、删除 (DELETE)，这四种类型和数据库里面的 binlog 概念保持一致。
- 而 Debezium 的数据结构，也有一个类似的元数据 op 字段， op 字段的取值也有四种，分别是 c、u、d、r，各自对应 create、update、delete、read。对于代表更新操作的 u，其数据部分同时包含了前镜像 (before) 和后镜像 (after)。

通过分析两种数据结构，Flink 和 Debezium 两者的底层数据是可以非常方便地对接起来的，因此Flink 做 CDC从技术上是非常合适的。

### 传统 CDC ETL 分析

来看下传统 CDC 的 ETL 分析链路，如下图所示：
![](https://ata2-img.oss-cn-zhangjiakou.aliyuncs.com/neweditor/35934362-2c5d-433c-b423-39f3a5288083.png#id=MGigI&originHeight=562&originWidth=1500&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

传统的基于 CDC 的 ETL 分析中，数据采集工具是必须的，国外用户常用 Debezium，国内用户常用阿里开源的 Canal，采集工具负责采集数据库的增量数据，一些采集工具也支持同步全量数据。采集到的数据一般输出到消息中间件如 Kafka，然后 Flink 计算引擎再去消费这一部分数据写入到目的端，目的端可以是各种 DB，数据湖，实时数仓和离线数仓。

注意，Flink 提供了 changelog-json format，可以将 changelog 数据写入离线数仓如 Hive / HDFS；对于实时数仓，Flink 支持将 changelog 通过 upsert-kafka connector 直接写入 Kafka。

是否可以使用 Flink CDC 去替换上图中虚线框内的采集组件和消息队列，从而简化分析链路，降低维护成本。同时更少的组件也意味着数据时效性能够进一步提高。答案是可以的，于是就有了基于 Flink CDC 的 ETL 分析流程。

### 基于 Flink CDC 的 ETL 分析

在使用了 Flink CDC 之后，除了组件更少，维护更方便外，另一个优势是通过 Flink SQL 极大地降低了用户使用门槛，可以看下面的例子：

![](https://ata2-img.oss-cn-zhangjiakou.aliyuncs.com/neweditor/06d644ae-1b6f-4571-96d9-35476f44cea2.png#id=sy8cS&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

该例子是通过 Flink CDC 去同步数据库数据并写入到 TiDB，用户直接使用 Flink SQL 创建了产品和订单的 MySQL-CDC 表，然后对数据流进行 JOIN 加工，加工后直接写入到下游数据库。通过一个 Flink SQL 作业就完成了 CDC 的数据分析，加工和同步。

大家会发现这是一个纯 SQL 作业，这意味着只要会 SQL 的 BI，业务线同学都可以完成此类工作。与此同时，用户也可以利用 Flink SQL 提供的丰富语法进行数据清洗、分析、聚合。

![](https://ata2-img.oss-cn-zhangjiakou.aliyuncs.com/neweditor/a0086ad9-2329-4a75-ac39-f9378ba17a77.png#id=Pxun6&originHeight=645&originWidth=1500&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

而这些能力，对于现有的 CDC 方案来说，进行数据的清洗，分析和聚合是非常困难的。

此外，利用 Flink SQL 双流 JOIN、维表 JOIN、UDTF 语法可以非常容易地完成数据打宽，以及各种业务逻辑加工。

![](https://ata2-img.oss-cn-zhangjiakou.aliyuncs.com/neweditor/0f68b43c-9f17-4601-ba98-23cf3a6dc3df.png#id=kycuj&originHeight=609&originWidth=1500&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

## Flink 1.x痛点

MySQL CDC 是 Flink CDC 中使用最多也是最重要的 Connector，在此基于 Flink CDC Connector 均为 MySQL CDC Connector。

随着 Flink CDC 项目的发展，得到了很多用户在社区的反馈，主要归纳为三个：

![](https://ata2-img.oss-cn-zhangjiakou.aliyuncs.com/neweditor/2fd421d4-2d13-4133-bc0a-380d7a8dab40.png#id=j3MKe&originHeight=437&originWidth=915&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

- 全量 + 增量读取的过程需要保证所有数据的一致性，因此需要通过加锁保证，但是加锁在数据库层面上是一个十分高危的操作。底层 Debezium 在保证数据一致性时，需要对读取的库或表加锁，全局锁可能导致数据库锁住，表级锁会锁住表的读，DBA 一般不给锁权限。
- 不支持水平扩展，因为 Flink CDC 底层是基于 Debezium，起架构是单节点，所以Flink CDC 只支持单并发。在全量阶段读取阶段，如果表非常大 (亿级别)，读取时间在小时甚至天级别，用户不能通过增加资源去提升作业速度。
- 全量读取阶段不支持 checkpoint：CDC 读取分为两个阶段，全量读取和增量读取，目前全量读取阶段是不支持 checkpoint 的，因此会存在一个问题：当我们同步全量数据时，假设需要 5 个小时，当我们同步了 4 小时的时候作业失败，这时候就需要重新开始，再读取 5 个小时。

### Flink CDC锁分析

Flink CDC 底层封装了 Debezium， Debezium 同步一张表分为两个阶段：

- 全量阶段：查询当前表中所有记录；
- 增量阶段：从 binlog 消费变更数据。

大部分用户使用的场景都是全量 + 增量同步，加锁是发生在全量阶段，目的是为了确定全量阶段的初始位点，保证增量 + 全量实现一条不多，一条不少，从而保证数据一致性。从下图中我们可以分析全局锁和表锁的一些加锁流程，左边红色线条是锁的生命周期，右边是 MySQL 开启可重复读事务的生命周期。

![](https://ata2-img.oss-cn-zhangjiakou.aliyuncs.com/neweditor/77c0b2ea-4b53-4253-8ddf-23ee7dd3d034.png#id=FmACl&originHeight=1068&originWidth=1166&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

以全局锁为例，首先是获取一个锁，然后再去开启可重复读的事务。这里锁住操作是读取 binlog 的起始位置和当前表的 schema。这样做的目的是保证 binlog 的起始位置和读取到的当前 schema 是可以对应上的，因为表的 schema 是会改变的，比如如删除列或者增加列。在读取这两个信息后，SnapshotReader 会在可重复读事务里读取全量数据，在全量数据读取完成后，会启动 BinlogReader 从读取的 binlog 起始位置开始增量读取，从而保证全量数据 + 增量数据的无缝衔接。

表锁是全局锁的退化版，因为全局锁的权限会比较高，因此在某些场景，用户只有表锁。表锁锁的时间会更长，因为表锁有个特征：锁提前释放了可重复读的事务默认会提交，所以锁需要等到全量数据读完后才能释放。

经过上面分析，接下来看看这些锁到底会造成怎样严重的后果：

![](https://ata2-img.oss-cn-zhangjiakou.aliyuncs.com/neweditor/977a0e29-404c-405f-8949-f8c0687e135c.png#id=bvP4A&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

link CDC 1.x 可以不加锁，能够满足大部分场景，但牺牲了一定的数据准确性。Flink CDC 1.x 默认加全局锁，虽然能保证数据一致性，但存在上述 hang 住数据的风险。

## Flink 2.0设计

通过上面的分析，可以知道 2.0 的设计方案，核心要解决上述的三个问题，即支持无锁、水平扩展、checkpoint。

DBlog 这篇论文里描述的无锁算法如下图所示：

![](https://ata2-img.oss-cn-zhangjiakou.aliyuncs.com/neweditor/4aa8da5f-8198-43db-8158-ecc85be09fea.png#id=VuRkK&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

### 核心逻辑

#### Chunk切分

左边是 Chunk 的切分算法描述，Chunk 的切分算法其实和很多数据库的分库分表原理类似，通过表的主键对表中的数据进行分片。假设每个 Chunk 的步长为 10，按照这个规则进行切分，只需要把这些 Chunk 的区间做成左开右闭或者左闭右开的区间，保证衔接后的区间能够等于表的主键区间即可。

右边是每个 Chunk 的无锁读算法描述，该算法的核心思想是在划分了 Chunk 后，对于每个 Chunk 的全量读取和增量读取，在不用锁的条件下完成一致性的合并。Chunk 的切分如下图所示：

![](https://ata2-img.oss-cn-zhangjiakou.aliyuncs.com/neweditor/1eb35ac3-bbcc-4d6b-a546-26118b13cc27.png#id=FGutM&originHeight=844&originWidth=1500&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

因为每个 chunk 只负责自己主键范围内的数据，不难推导，只要能够保证每个 Chunk 读取的一致性，就能保证整张表读取的一致性，这便是无锁算法的基本原理。

#### Chunk分配（实现并行读取数据&CheckPoint）

![](https://ata2-img.oss-cn-zhangjiakou.aliyuncs.com/neweditor/75cb4653-3458-43a5-8ea2-6b3b9ca30aa9.png#id=BjfJH&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

上图描述的是单个 Chunk 的一致性读，但是如果有多个表分了很多不同的 Chunk，且这些 Chunk 分发到了不同的 task 中，那么如何分发 Chunk 并保证全局一致性读呢？

这个就是基于 FLIP-27 来优雅地实现的，通过下图可以看到有 SourceEnumerator 的组件，这个组件主要用于 Chunk 的划分，划分好的 Chunk 会提供给下游的 SourceReader 去读取，通过把 chunk 分发给不同的 SourceReader 便实现了并发读取 Snapshot Chunk 的过程，同时基于 FLIP-27 我们能较为方便地做到 chunk 粒度的 checkpoint。

将划分好的Chunk分发给多个 SourceReader，每个SourceReader读取表中的一部分数据，实现了并行读取的目标。

同时在每个Chunk读取的时候可以单独做CheckPoint，某个Chunk读取失败只需要单独执行该Chunk的任务，而不需要像1.x中失败了只能从头读取。

若每个SourceReader保证了数据一致性，则全表就保证了数据一致性。

#### Chunk读取（实现无锁读取）

读取可以分为5个阶段
1）SourceReader读取表数据之前先记录当前的Binlog位置信息记为低位点；
2）SourceReader将自身区间内的数据查询出来并放置在buffer中；
3）查询完成之后记录当前的Binlog位置信息记为高位点；
4）在增量部分消费从低位点到高位点的Binlog；
5）根据主键，对buffer中的数据进行修正并输出。

通过以上5个阶段可以保证每个Chunk最终的输出就是在高位点时该Chunk中最新的数据，但是目前只是做到了保证单个Chunk中的数据一致性。

Netflix 的 DBLog 论文中 Chunk 读取算法是通过在 DB 维护一张信号表，再通过信号表在 binlog 文件中打点，记录每个 chunk 读取前的 Low Position (低位点) 和读取结束之后 High Position (高位点) ，在低位点和高位点之间去查询该 Chunk 的全量数据。在读取出这一部分 Chunk 的数据之后，再将这 2 个位点之间的 binlog 增量数据合并到 chunk 所属的全量数据，从而得到高位点时刻，该 chunk 对应的全量数据。

Flink CDC 结合自身的情况，在 Chunk 读取算法上做了去信号表的改进，不需要额外维护信号表，通过直接读取 binlog 位点替代在 binlog 中做标记的功能，整体的 chunk 读算法描述如下图所示：

![](https://ata2-img.oss-cn-zhangjiakou.aliyuncs.com/neweditor/57d79dda-db1d-4e05-b2eb-3020af25c2d3.png#id=tSqwO&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

比如正在读取 Chunk-1，Chunk 的区间是 [K1, K10]，首先直接将该区间内的数据 select 出来并把它存在 buffer 中，在 select 之前记录 binlog 的一个位点 (低位点)，select 完成后记录 binlog 的一个位点 (高位点)。然后开始增量部分，消费从低位点到高位点的 binlog。

- 图中的 - ( k2,100 ) + ( k2,108 ) 记录表示这条数据的值从 100 更新到 108；
- 第二条记录是删除 k3；
- 第三条记录是更新 k2 为 119；
- 第四条记录是 k5 的数据由原来的 77 变更为 100。

观察图片中右下角最终的输出，会发现在消费该 chunk 的 binlog 时，出现的 key 是k2、k3、k5，我们前往 buffer 将这些 key 做标记。

*对于 k1、k4、k6、k7 来说，在高位点读取完毕之后，这些记录没有变化过，所以这些数据是可以直接输出的；

- 对于改变过的数据，则需要将增量的数据合并到全量的数据中，只保留合并后的最终数据。例如，k2 最终的结果是 119 ，那么只需要输出 +(k2,119)，而不需要中间发生过改变的数据。

通过这种方式，Chunk 最终的输出就是在高位点是 chunk 中最新的数据。

#### Chunk汇报

当 Snapshot Chunk 读取完成之后，需要有一个汇报的流程，如下图中橘色的汇报信息，将 Snapshot Chunk 完成信息汇报给 SourceEnumerator。

![](https://ata2-img.oss-cn-zhangjiakou.aliyuncs.com/neweditor/bc456f82-3036-462a-898d-fff0e690873c.png#id=b18tf&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

汇报的主要目的是为了后续分发 binlog chunk (如下图)。因为 Flink CDC 支持全量 + 增量同步，所以当所有 Snapshot Chunk 读取完成之后，还需要消费增量的 binlog，这是通过下发一个 binlog chunk 给任意一个 Source Reader 进行单并发读取实现的。

#### Chunk分配

FlinkCDC是支持全量+增量数据同步的，在SourceEnumerator接收到所有的Snapshot Chunk完成信息之后，还有一个消费增量数据（Binlog）的任务，此时是通过下发Binlog Chunk给任意一个SourceReader进行单并发读取来实现的。

![](https://ata2-img.oss-cn-zhangjiakou.aliyuncs.com/neweditor/98daa1c4-7263-42aa-b1f1-c7ea9e47a770.png#id=ivqSK&originHeight=818&originWidth=1368&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

整体流程可以概括为，首先通过主键对表进行 Snapshot Chunk 划分，再将 Snapshot Chunk 分发给多个 SourceReader，每个 Snapshot Chunk 读取时通过算法实现无锁条件下的一致性读，SourceReader 读取时支持 chunk 粒度的 checkpoint，在所有 Snapshot Chunk 读取完成后，下发一个 binlog chunk 进行增量部分的 binlog 读取，这便是 Flink CDC 2.0 的整体流程，如下图所示：

![](https://ata2-img.oss-cn-zhangjiakou.aliyuncs.com/neweditor/1c49a9f8-5e87-46da-a6c1-f4f484c59773.png#id=UNuQG&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

Flink CDC 2.0 的核心改进和提升包括：

- 提供 MySQL CDC 2.0，核心feature 包括 
   - 并发读取，全量数据的读取性能可以水平扩展；
   - 全程无锁，不对线上业务产生锁的风险；
   - 断点续传，支持全量阶段的 checkpoint。

### 关键代码

#### 根据物理主键将table切分成chunks

```java
    /**
     * We can use evenly-sized chunks or unevenly-sized chunks when split table into chunks, using
     * evenly-sized chunks which is much efficient, using unevenly-sized chunks which will request
     * many queries and is not efficient.
     */
    private List<ChunkRange> splitTableIntoChunks(
            JdbcConnection jdbc, TableId tableId, Column splitColumn) throws SQLException {
        final String splitColumnName = splitColumn.name();
        final Object[] minMaxOfSplitColumn = queryMinMax(jdbc, tableId, splitColumnName);
        final Object min = minMaxOfSplitColumn[0];
        final Object max = minMaxOfSplitColumn[1];
        if (min == null || max == null || min.equals(max)) {
            // empty table, or only one row, return full table scan as a chunk
            return Collections.singletonList(ChunkRange.all());
        }

        final int chunkSize = sourceConfig.getSplitSize();
        final double distributionFactorUpper = sourceConfig.getDistributionFactorUpper();
        final double distributionFactorLower = sourceConfig.getDistributionFactorLower();

        if (isEvenlySplitColumn(splitColumn)) {
            long approximateRowCnt = queryApproximateRowCnt(jdbc, tableId);
            double distributionFactor =
                    calculateDistributionFactor(tableId, min, max, approximateRowCnt);

            boolean dataIsEvenlyDistributed =
                    doubleCompare(distributionFactor, distributionFactorLower) >= 0
                            && doubleCompare(distributionFactor, distributionFactorUpper) <= 0;

            if (dataIsEvenlyDistributed) {
                // the minimum dynamic chunk size is at least 1
                final int dynamicChunkSize = Math.max((int) (distributionFactor * chunkSize), 1);
                return splitEvenlySizedChunks(
                        tableId, min, max, approximateRowCnt, chunkSize, dynamicChunkSize);
            } else {
                return splitUnevenlySizedChunks(
                        jdbc, tableId, splitColumnName, min, max, chunkSize);
            }
        } else {
            return splitUnevenlySizedChunks(jdbc, tableId, splitColumnName, min, max, chunkSize);
        }
    }
```

#### Chunk数据读取

主要涉及到三个步骤，低水位点设置，chunk数据查询，高水位点设置。

```java
    @Override
    protected SnapshotResult<MySqlOffsetContext> doExecute(
            ChangeEventSourceContext context,
            MySqlOffsetContext previousOffset,
            SnapshotContext<MySqlOffsetContext> snapshotContext,
            SnapshottingTask snapshottingTask)
            throws Exception {
        final RelationalSnapshotChangeEventSource.RelationalSnapshotContext<MySqlOffsetContext>
                ctx =
                        (RelationalSnapshotChangeEventSource.RelationalSnapshotContext<
                                        MySqlOffsetContext>)
                                snapshotContext;
        ctx.offset = previousOffset;
        final SignalEventDispatcher signalEventDispatcher =
                new SignalEventDispatcher(
                        previousOffset.getPartition(),
                        topicSelector.topicNameFor(snapshotSplit.getTableId()),
                        dispatcher.getQueue());
    	// 获取当前binlog的offset，设置为low watermark
        final BinlogOffset lowWatermark = currentBinlogOffset(jdbcConnection);
        LOG.info(
                "Snapshot step 1 - Determining low watermark {} for split {}",
                lowWatermark,
                snapshotSplit);
        ((SnapshotSplitReader.SnapshotSplitChangeEventSourceContextImpl) (context))
                .setLowWatermark(lowWatermark);
        signalEventDispatcher.dispatchWatermarkEvent(
                snapshotSplit, lowWatermark, SignalEventDispatcher.WatermarkKind.LOW);

        LOG.info("Snapshot step 2 - Snapshotting data");
        // 查询chunk数据并且分发数据变更事件
        createDataEvents(ctx, snapshotSplit.getTableId());

        // 设置high watermark
        final BinlogOffset highWatermark = currentBinlogOffset(jdbcConnection);
        LOG.info(
                "Snapshot step 3 - Determining high watermark {} for split {}",
                highWatermark,
                snapshotSplit);
        signalEventDispatcher.dispatchWatermarkEvent(
                snapshotSplit, highWatermark, SignalEventDispatcher.WatermarkKind.HIGH);
        ((SnapshotSplitReader.SnapshotSplitChangeEventSourceContextImpl) (context))
                .setHighWatermark(highWatermark);

        return SnapshotResult.completed(ctx.offset);
    }
```

#### 增量读取

在完整阶段读取所有拆分后，MySqlHybridSplitAssigner 创建一个 BinlogSplit 用于后续增量读取。 创建 BinlogSplit 时，会从所有已完成的完整拆分中过滤出最小的 BinlogOffset。

```java
    private MySqlBinlogSplit createBinlogSplit() {
        final List<MySqlSnapshotSplit> assignedSnapshotSplit =
                snapshotSplitAssigner.getAssignedSplits().values().stream()
                        .sorted(Comparator.comparing(MySqlSplit::splitId))
                        .collect(Collectors.toList());

        Map<String, BinlogOffset> splitFinishedOffsets =
                snapshotSplitAssigner.getSplitFinishedOffsets();
        final List<FinishedSnapshotSplitInfo> finishedSnapshotSplitInfos = new ArrayList<>();

        BinlogOffset minBinlogOffset = null;
        for (MySqlSnapshotSplit split : assignedSnapshotSplit) {
            // find the min binlog offset
            BinlogOffset binlogOffset = splitFinishedOffsets.get(split.splitId());
            if (minBinlogOffset == null || binlogOffset.isBefore(minBinlogOffset)) {
                minBinlogOffset = binlogOffset;
            }
            finishedSnapshotSplitInfos.add(
                    new FinishedSnapshotSplitInfo(
                            split.getTableId(),
                            split.splitId(),
                            split.getSplitStart(),
                            split.getSplitEnd(),
                            binlogOffset));
        }

        // the finishedSnapshotSplitInfos is too large for transmission, divide it to groups and
        // then transfer them

        boolean divideMetaToGroups = finishedSnapshotSplitInfos.size() > splitMetaGroupSize;
        return new MySqlBinlogSplit(
                BINLOG_SPLIT_ID,
                minBinlogOffset == null ? BinlogOffset.INITIAL_OFFSET : minBinlogOffset,
                BinlogOffset.NO_STOPPING_OFFSET,
                divideMetaToGroups ? new ArrayList<>() : finishedSnapshotSplitInfos,
                new HashMap<>(),
                finishedSnapshotSplitInfos.size());
    }
```

#### 根据高低位点数据对全量数据进行修正

```java
    public Iterator<SourceRecord> pollSplitRecords() throws InterruptedException {
        checkReadException();

        if (hasNextElement.get()) {
            // data input: [low watermark event][snapshot events][high watermark event][binlog
            // events][binlog-end event]
            // data output: [low watermark event][normalized events][high watermark event]
            boolean reachBinlogStart = false;
            boolean reachBinlogEnd = false;
            SourceRecord lowWatermark = null;
            SourceRecord highWatermark = null;
            Map<Struct, SourceRecord> snapshotRecords = new HashMap<>();
            while (!reachBinlogEnd) {
                checkReadException();
                List<DataChangeEvent> batch = queue.poll();
                for (DataChangeEvent event : batch) {
                    SourceRecord record = event.getRecord();
                    if (lowWatermark == null) {
                        lowWatermark = record;
                        // 检查第一条record是不是低水位信号事件，如果是继续循环，否则抛出异常
                        assertLowWatermark(lowWatermark);
                        continue;
                    }
                	// 寻找高水位点，如果找到，则表示快照读取完毕，开始读取Binlog日志
                    if (highWatermark == null && isHighWatermarkEvent(record)) {
                        highWatermark = record;
                        // snapshot events capture end and begin to capture binlog events
                        reachBinlogStart = true;
                        continue;
                    }

                    // 如果找到BINLOG_END位点，结束循环，表示一个完整的data input
                    if (reachBinlogStart && RecordUtils.isEndWatermarkEvent(record)) {
                        // capture to end watermark events, stop the loop
                        reachBinlogEnd = true;
                        break;
                    }

                    if (!reachBinlogStart) {
                        // 如果没有达到High Watermark，则继续读取chunk数据
                        snapshotRecords.put((Struct) record.key(), record);
                    } else {
                        // 如果达到了High Watermark，但是没有达到BINLOG_END，则将binlog日志合并到chunk数据中
                        if (isRequiredBinlogRecord(record)) {
                            // upsert binlog events through the record key
                            upsertBinlog(snapshotRecords, record);
                        }
                    }
                }
            }
            // snapshot split return its data once
            hasNextElement.set(false);

            final List<SourceRecord> normalizedRecords = new ArrayList<>();
            normalizedRecords.add(lowWatermark);
            normalizedRecords.addAll(formatMessageTimestamp(snapshotRecords.values()));
            normalizedRecords.add(highWatermark);
            return normalizedRecords.iterator();
        }
        // the data has been polled, no more data
        reachEnd.compareAndSet(false, true);
        return null;
    }
```

#### 读取低位点到高位点之间的Binlog

```java
    /**
     * Returns the record should emit or not.
     *
     * <p>The watermark signal algorithm is the binlog split reader only sends the binlog event that
     * belongs to its finished snapshot splits. For each snapshot split, the binlog event is valid
     * since the offset is after its high watermark.
     *
     * 
<pre> E.g: the data input is :
     *    snapshot-split-0 info : [0,    1024) highWatermark0
     *    snapshot-split-1 info : [1024, 2048) highWatermark1
     *  the data output is:
     *  only the binlog event belong to [0,    1024) and offset is after highWatermark0 should send,
     *  only the binlog event belong to [1024, 2048) and offset is after highWatermark1 should send.
     * </pre>
*/
    private boolean shouldEmit(SourceRecord sourceRecord) {
        if (isDataChangeRecord(sourceRecord)) {
            TableId tableId = getTableId(sourceRecord);
            BinlogOffset position = getBinlogPosition(sourceRecord);
            if (hasEnterPureBinlogPhase(tableId, position)) {
                return true;
            }
            // only the table who captured snapshot splits need to filter
            if (finishedSplitsInfo.containsKey(tableId)) {
                RowType splitKeyType =
                        ChunkUtils.getChunkKeyColumnType(
                                statefulTaskContext.getDatabaseSchema().tableFor(tableId),
                                statefulTaskContext.getSourceConfig().getChunkKeyColumn());
                Object[] key =
                        getSplitKey(
                                splitKeyType,
                                sourceRecord,
                                statefulTaskContext.getSchemaNameAdjuster());
                for (FinishedSnapshotSplitInfo splitInfo : finishedSplitsInfo.get(tableId)) {
                    if (RecordUtils.splitKeyRangeContains(
                                    key, splitInfo.getSplitStart(), splitInfo.getSplitEnd())
                            && position.isAfter(splitInfo.getHighWatermark())) {
                        return true;
                    }
                }
            }
            // not in the monitored splits scope, do not emit
            return false;
        }
        // always send the schema change event and signal event
        // we need record them to state of Flink
        return true;
    }
```

## Flink CDC quick start

### Usage for Table/SQL API

1. Create the MySQL user root

```sql
mysql> CREATE USER 'root'@'localhost' IDENTIFIED BY 'root';
```

2. Grant regard permission to the user

```sql
mysql> GRANT SELECT, SHOW DATABASES, REPLICATION SLAVE, REPLICATION CLIENT ON *.* TO 'root' IDENTIFIED BY 'root';
```

3. Config the bin log

```xml
# log_bin
max_allowed_packet=1024M
server_id=1
log-bin=master
binlog_format=row
binlog-do-db=gmall
```

4. Data Synchronization

```java
public class FlinkCdcMysql {
    public static void main(String[] args) throws Exception {
        // Create execution environment
        StreamExecutionEnvironment executionEnvironment = StreamExecutionEnvironment.getExecutionEnvironment();
        executionEnvironment.setParallelism(1);

        StreamTableEnvironment streamTableEnvironment = StreamTableEnvironment.create(executionEnvironment);

        streamTableEnvironment.executeSql("CREATE TABLE product (\n" +
            "    id INT primary key,\n" +
            "    name STRING" +
            ") WITH (\n" +
            "          'connector' = 'mysql-cdc',\n" +
            "          'hostname' = 'localhost',\n" +
            "          'port' = '3312',\n" +
            "          'username' = 'root',\n" +
            "          'password' = 'root',\n" +
            "          'database-name' = 'mall',\n" +
            "          'table-name' = 'product'," +
            // Full data and incremental data synchronization.
            "          'scan.startup.mode' = 'initial'      " +
            ")");

        Table table = streamTableEnvironment.sqlQuery("select * from product");
        DataStream<Tuple2<Boolean, Row>> reactStream = streamTableEnvironment.toRetractStream(table, Row.class);
        reactStream.print();

        executionEnvironment.execute("Flink CDC");

    }

}
```

5. 输出格式，输出的格式比较固定简洁，适合直接拿来使用

```xml
(true,+I[1, Apple])
(true,+I[2, Aubergine])
(true,+I[3, Cucumber])
(true,+I[4, Orange])
```

### Usage for DataStream API

1. Example code

```java
public class FlinkCdcDataStream {
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment streamExecutionEnvironment = StreamExecutionEnvironment.getExecutionEnvironment();
        streamExecutionEnvironment.setParallelism(1);

        Properties properties = new Properties();
        // do not use lock
        properties.put("snapshot.locking.mode", "none");
        MySqlSource<String> debeziumSourceFunction = MySqlSource.<String>builder().hostname("localhost")
            .port(3312).username("root").password("root").databaseList("mall").tableList("mall.product").deserializer(
                new StringDebeziumDeserializationSchema()).startupOptions(StartupOptions.initial()).debeziumProperties(
                properties).build();

        streamExecutionEnvironment.fromSource(debeziumSourceFunction, WatermarkStrategy.noWatermarks(),
            "Flink CDC MySQL source").setParallelism(1).print();
        streamExecutionEnvironment.execute("Flink CDC DataStream");
    }
}
```

2. 输出格式，输出的是类似日志格式，更灵活

```json
SourceRecord{sourcePartition={server=mysql_binlog_source}, sourceOffset={transaction_id=null, ts_sec=1661047800, file=, pos=0}} ConnectRecord{topic='mysql_binlog_source.mall.product', kafkaPartition=null, key=Struct{id=1}, keySchema=Schema{mysql_binlog_source.mall.product.Key:STRUCT}, value=Struct{after=Struct{id=1,name=Apple},source=Struct{version=1.5.4.Final,connector=mysql,name=mysql_binlog_source,ts_ms=0,db=mall,table=product,server_id=0,file=,pos=0,row=0},op=r,ts_ms=1661047800940}, valueSchema=Schema{mysql_binlog_source.mall.product.Envelope:STRUCT}, timestamp=null, headers=ConnectHeaders(headers=)}
```

## Flink CDC未来规划

关于 CDC 项目的未来规划，围绕稳定性，进阶 feature 和生态集成三个方面展开。

- 稳定性 
   - 通过社区的方式吸引更多的开发者，公司的开源力量提升 Flink CDC 的成熟度；
   - 支持 Lazy Assigning。Lazy Assigning 的思路是将 chunk 先划分一批，而不是一次性进行全部划分。当前 Source Reader 对数据读取进行分片是一次性全部划分好所有 chunk，例如有 1 万个 chunk，可以先划分 1 千个 chunk，而不是一次性全部划分，在 SourceReader 读取完 1 千 chunk 后再继续划分，节约划分 chunk 的时间。
- 进阶 Feature 
   - 支持 Schema Evolution。这个场景是：当同步数据库的过程中，突然在表中添加了一个字段，并且希望后续同步下游系统的时候能够自动加入这个字段；
   - 支持 Watermark Pushdown 通过 CDC 的 binlog 获取到一些心跳信息，这些心跳的信息可以作为一个 Watermark，通过这个心跳信息可以知道到这个流当前消费的一些进度；
   - 支持 META 数据，分库分表的场景下，有可能需要元数据知道这条数据来源哪个库哪个表，在下游系统入湖入仓可以有更多的灵活操作；
   - 整库同步：用户要同步整个数据库只需一行 SQL 语法即可完成，而不用每张表定义一个 DDL 和 query
- 生态集成 
   - 集成更多上游数据库，如 Oracle，MS SqlServer。Cloudera 目前正在积极贡献 oracle-cdc connector；
   - 在入湖层面，Hudi 和 Iceberg 写入上有一定的优化空间，例如在高 QPS 入湖的时候，数据分布有比较大的性能影响，这一点可以通过与生态打通和集成继续优化。

## Reference

[DBLog: A Watermark Based Change-Data-Capture Framework](https://arxiv.org/abs/2010.12597)
[Introducing backup locks in Percona Server - Percona Database Performance Blog](https://www.percona.com/blog/2014/03/11/introducing-backup-locks-percona-server-2/?spm=a2csy.flink.0.0.55796c2eBN3ubd)
[FLIP-27: Refactor Source Interface - Apache Flink - Apache Software Foundation](https://cwiki.apache.org/confluence/display/FLINK/FLIP-27%3A+Refactor+Source+Interface?spm=a2csy.flink.0.0.55796c2eBN3ub)
