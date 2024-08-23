<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>
<tagfile doxygen_version="1.9.8">
  <compound kind="file">
    <name>choose_classic_markers.hpp</name>
    <path>singlepp/</path>
    <filename>choose__classic__markers_8hpp.html</filename>
    <class kind="struct">singlepp::ChooseClassicMarkersOptions</class>
    <namespace>singlepp</namespace>
  </compound>
  <compound kind="file">
    <name>classify_integrated.hpp</name>
    <path>singlepp/</path>
    <filename>classify__integrated_8hpp.html</filename>
    <class kind="struct">singlepp::ClassifyIntegratedOptions</class>
    <class kind="struct">singlepp::ClassifyIntegratedBuffers</class>
    <class kind="struct">singlepp::ClassifyIntegratedResults</class>
    <namespace>singlepp</namespace>
  </compound>
  <compound kind="file">
    <name>classify_single.hpp</name>
    <path>singlepp/</path>
    <filename>classify__single_8hpp.html</filename>
    <class kind="struct">singlepp::ClassifySingleOptions</class>
    <class kind="struct">singlepp::ClassifySingleBuffers</class>
    <class kind="struct">singlepp::ClassifySingleResults</class>
    <namespace>singlepp</namespace>
  </compound>
  <compound kind="file">
    <name>Intersection.hpp</name>
    <path>singlepp/</path>
    <filename>Intersection_8hpp.html</filename>
    <namespace>singlepp</namespace>
  </compound>
  <compound kind="file">
    <name>Markers.hpp</name>
    <path>singlepp/</path>
    <filename>Markers_8hpp.html</filename>
    <namespace>singlepp</namespace>
  </compound>
  <compound kind="file">
    <name>singlepp.hpp</name>
    <path>singlepp/</path>
    <filename>singlepp_8hpp.html</filename>
    <namespace>singlepp</namespace>
  </compound>
  <compound kind="file">
    <name>train_integrated.hpp</name>
    <path>singlepp/</path>
    <filename>train__integrated_8hpp.html</filename>
    <class kind="struct">singlepp::TrainIntegratedInput</class>
    <class kind="class">singlepp::TrainedIntegrated</class>
    <class kind="struct">singlepp::TrainIntegratedOptions</class>
    <namespace>singlepp</namespace>
  </compound>
  <compound kind="file">
    <name>train_single.hpp</name>
    <path>singlepp/</path>
    <filename>train__single_8hpp.html</filename>
    <class kind="struct">singlepp::TrainSingleOptions</class>
    <class kind="class">singlepp::TrainedSingle</class>
    <class kind="class">singlepp::TrainedSingleIntersect</class>
    <namespace>singlepp</namespace>
  </compound>
  <compound kind="struct">
    <name>singlepp::ChooseClassicMarkersOptions</name>
    <filename>structsinglepp_1_1ChooseClassicMarkersOptions.html</filename>
    <member kind="variable">
      <type>int</type>
      <name>number</name>
      <anchorfile>structsinglepp_1_1ChooseClassicMarkersOptions.html</anchorfile>
      <anchor>a9d44e81aaf6d22c6d32264281e548ffd</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>int</type>
      <name>num_threads</name>
      <anchorfile>structsinglepp_1_1ChooseClassicMarkersOptions.html</anchorfile>
      <anchor>a7c694f5f2e2cf5d8610aa3028bce4055</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>singlepp::ClassifyIntegratedBuffers</name>
    <filename>structsinglepp_1_1ClassifyIntegratedBuffers.html</filename>
    <templarg>typename RefLabel_</templarg>
    <templarg>typename Float_</templarg>
    <member kind="variable">
      <type>RefLabel_ *</type>
      <name>best</name>
      <anchorfile>structsinglepp_1_1ClassifyIntegratedBuffers.html</anchorfile>
      <anchor>a51dd05dee04d9d73f1ad6d62655ca1e1</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::vector&lt; Float_ * &gt;</type>
      <name>scores</name>
      <anchorfile>structsinglepp_1_1ClassifyIntegratedBuffers.html</anchorfile>
      <anchor>a947caf925fb5a04bbfe67de950549c6e</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>Float_ *</type>
      <name>delta</name>
      <anchorfile>structsinglepp_1_1ClassifyIntegratedBuffers.html</anchorfile>
      <anchor>a7307840e67b7743f985ad4c780090168</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>singlepp::ClassifyIntegratedOptions</name>
    <filename>structsinglepp_1_1ClassifyIntegratedOptions.html</filename>
    <templarg>typename Float_</templarg>
    <member kind="variable">
      <type>Float_</type>
      <name>quantile</name>
      <anchorfile>structsinglepp_1_1ClassifyIntegratedOptions.html</anchorfile>
      <anchor>a6d4d11059b19ca5066ceed0d9e36bf19</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>int</type>
      <name>num_threads</name>
      <anchorfile>structsinglepp_1_1ClassifyIntegratedOptions.html</anchorfile>
      <anchor>a5562f346d14d824da25e56e367194c7c</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>singlepp::ClassifyIntegratedResults</name>
    <filename>structsinglepp_1_1ClassifyIntegratedResults.html</filename>
    <templarg>typename RefLabel_</templarg>
    <templarg>typename Float_</templarg>
    <member kind="variable">
      <type>std::vector&lt; RefLabel_ &gt;</type>
      <name>best</name>
      <anchorfile>structsinglepp_1_1ClassifyIntegratedResults.html</anchorfile>
      <anchor>a2b2ab0c6b6cbefb29acbd685801f1e34</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::vector&lt; std::vector&lt; Float_ &gt; &gt;</type>
      <name>scores</name>
      <anchorfile>structsinglepp_1_1ClassifyIntegratedResults.html</anchorfile>
      <anchor>a104a0cf9a54e105f9e400f7b21306c81</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::vector&lt; Float_ &gt;</type>
      <name>delta</name>
      <anchorfile>structsinglepp_1_1ClassifyIntegratedResults.html</anchorfile>
      <anchor>a7fa791e370428f1535d3218fd3313987</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>singlepp::ClassifySingleBuffers</name>
    <filename>structsinglepp_1_1ClassifySingleBuffers.html</filename>
    <templarg>typename Label_</templarg>
    <templarg>typename Float_</templarg>
    <member kind="variable">
      <type>Label_ *</type>
      <name>best</name>
      <anchorfile>structsinglepp_1_1ClassifySingleBuffers.html</anchorfile>
      <anchor>a051c296c60087f2bfa9c46240c885487</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::vector&lt; Float_ * &gt;</type>
      <name>scores</name>
      <anchorfile>structsinglepp_1_1ClassifySingleBuffers.html</anchorfile>
      <anchor>a77d83e65579276b44bf9b513c6a7e0e8</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>Float_ *</type>
      <name>delta</name>
      <anchorfile>structsinglepp_1_1ClassifySingleBuffers.html</anchorfile>
      <anchor>a044d3e44f4d19f387a62a2ba9f0b87a3</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>singlepp::ClassifySingleOptions</name>
    <filename>structsinglepp_1_1ClassifySingleOptions.html</filename>
    <templarg>typename Float_</templarg>
    <member kind="variable">
      <type>Float_</type>
      <name>quantile</name>
      <anchorfile>structsinglepp_1_1ClassifySingleOptions.html</anchorfile>
      <anchor>aebeadb5d5c853a7e2d0c6d9d7f611656</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>Float_</type>
      <name>fine_tune_threshold</name>
      <anchorfile>structsinglepp_1_1ClassifySingleOptions.html</anchorfile>
      <anchor>ac4b20b4e0d9f50695d60f86dc2f48406</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>fine_tune</name>
      <anchorfile>structsinglepp_1_1ClassifySingleOptions.html</anchorfile>
      <anchor>a2e9ca1a6da5fcb8877d9029db4b46b7a</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>int</type>
      <name>num_threads</name>
      <anchorfile>structsinglepp_1_1ClassifySingleOptions.html</anchorfile>
      <anchor>ad8d08a1478387fce71cf319c234aba8c</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>singlepp::ClassifySingleResults</name>
    <filename>structsinglepp_1_1ClassifySingleResults.html</filename>
    <templarg>typename Label_</templarg>
    <templarg>typename Float_</templarg>
    <member kind="variable">
      <type>std::vector&lt; Label_ &gt;</type>
      <name>best</name>
      <anchorfile>structsinglepp_1_1ClassifySingleResults.html</anchorfile>
      <anchor>a85a0c24158c546a629455dabd24d5cb8</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::vector&lt; std::vector&lt; Float_ &gt; &gt;</type>
      <name>scores</name>
      <anchorfile>structsinglepp_1_1ClassifySingleResults.html</anchorfile>
      <anchor>a9d75ebe7e597aa2048c2ebc4cc3e4a29</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::vector&lt; Float_ &gt;</type>
      <name>delta</name>
      <anchorfile>structsinglepp_1_1ClassifySingleResults.html</anchorfile>
      <anchor>a095f534120dc3cc11de7b58a2eef64f1</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>singlepp::TrainedIntegrated</name>
    <filename>classsinglepp_1_1TrainedIntegrated.html</filename>
    <templarg>typename Index_</templarg>
    <member kind="function">
      <type>size_t</type>
      <name>num_references</name>
      <anchorfile>classsinglepp_1_1TrainedIntegrated.html</anchorfile>
      <anchor>ad71b65ed97da76741dacc690247b1239</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>size_t</type>
      <name>num_labels</name>
      <anchorfile>classsinglepp_1_1TrainedIntegrated.html</anchorfile>
      <anchor>a1e143e6103e7f64273ab6050c7ad4e4f</anchor>
      <arglist>(size_t r) const</arglist>
    </member>
    <member kind="function">
      <type>size_t</type>
      <name>num_profiles</name>
      <anchorfile>classsinglepp_1_1TrainedIntegrated.html</anchorfile>
      <anchor>a7a79785c99f5cd71388a3fe020171e38</anchor>
      <arglist>(size_t r) const</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>singlepp::TrainedSingle</name>
    <filename>classsinglepp_1_1TrainedSingle.html</filename>
    <templarg>typename Index_</templarg>
    <templarg>typename Float_</templarg>
    <member kind="function">
      <type>const Markers&lt; Index_ &gt; &amp;</type>
      <name>get_markers</name>
      <anchorfile>classsinglepp_1_1TrainedSingle.html</anchorfile>
      <anchor>af950945f130f3d957fc06bf029117a9e</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>const std::vector&lt; Index_ &gt; &amp;</type>
      <name>get_subset</name>
      <anchorfile>classsinglepp_1_1TrainedSingle.html</anchorfile>
      <anchor>a2054db73c1dfa8cff3024b9a89d8c304</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>size_t</type>
      <name>num_labels</name>
      <anchorfile>classsinglepp_1_1TrainedSingle.html</anchorfile>
      <anchor>a8b6ede4f1878f128f9af840e0f0984d0</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>size_t</type>
      <name>num_profiles</name>
      <anchorfile>classsinglepp_1_1TrainedSingle.html</anchorfile>
      <anchor>ae9650051a7b4e54584670bb64bcf600e</anchor>
      <arglist>() const</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>singlepp::TrainedSingleIntersect</name>
    <filename>classsinglepp_1_1TrainedSingleIntersect.html</filename>
    <templarg>typename Index_</templarg>
    <templarg>typename Float_</templarg>
    <member kind="function">
      <type>const Markers&lt; Index_ &gt; &amp;</type>
      <name>get_markers</name>
      <anchorfile>classsinglepp_1_1TrainedSingleIntersect.html</anchorfile>
      <anchor>a6b8e1e521a826f67858f0c97c99c047e</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>const std::vector&lt; Index_ &gt; &amp;</type>
      <name>get_test_subset</name>
      <anchorfile>classsinglepp_1_1TrainedSingleIntersect.html</anchorfile>
      <anchor>a82b8ce08c57b8dc6bbb2acd0a10ea9d7</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>const std::vector&lt; Index_ &gt; &amp;</type>
      <name>get_ref_subset</name>
      <anchorfile>classsinglepp_1_1TrainedSingleIntersect.html</anchorfile>
      <anchor>ac2ec2b035a62da9b175f84a37486d1bf</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>size_t</type>
      <name>num_labels</name>
      <anchorfile>classsinglepp_1_1TrainedSingleIntersect.html</anchorfile>
      <anchor>add601cf587ac70be43e69abf501197fe</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>size_t</type>
      <name>num_profiles</name>
      <anchorfile>classsinglepp_1_1TrainedSingleIntersect.html</anchorfile>
      <anchor>ae66e9daad1171ea3e28951a4974c292b</anchor>
      <arglist>() const</arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>singlepp::TrainIntegratedInput</name>
    <filename>structsinglepp_1_1TrainIntegratedInput.html</filename>
    <templarg>typename Value_</templarg>
    <templarg>typename Index_</templarg>
    <templarg>typename Label_</templarg>
  </compound>
  <compound kind="struct">
    <name>singlepp::TrainIntegratedOptions</name>
    <filename>structsinglepp_1_1TrainIntegratedOptions.html</filename>
    <member kind="variable">
      <type>int</type>
      <name>num_threads</name>
      <anchorfile>structsinglepp_1_1TrainIntegratedOptions.html</anchorfile>
      <anchor>ae500e9838e046e9f191057e6eb7f6803</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>singlepp::TrainSingleOptions</name>
    <filename>structsinglepp_1_1TrainSingleOptions.html</filename>
    <templarg>typename Index_</templarg>
    <templarg>typename Float_</templarg>
    <member kind="variable">
      <type>int</type>
      <name>top</name>
      <anchorfile>structsinglepp_1_1TrainSingleOptions.html</anchorfile>
      <anchor>aa72f73c9937301bd90c2fec19f523479</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>std::shared_ptr&lt; knncolle::Builder&lt; knncolle::SimpleMatrix&lt; Index_, Index_, Float_ &gt;, Float_ &gt; &gt;</type>
      <name>trainer</name>
      <anchorfile>structsinglepp_1_1TrainSingleOptions.html</anchorfile>
      <anchor>a36a62c8abfc04a46be71e2ba9e179a60</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>int</type>
      <name>num_threads</name>
      <anchorfile>structsinglepp_1_1TrainSingleOptions.html</anchorfile>
      <anchor>a23ecc04e6d269ad1ede160c7838b8a63</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="namespace">
    <name>singlepp</name>
    <filename>namespacesinglepp.html</filename>
    <class kind="struct">singlepp::ChooseClassicMarkersOptions</class>
    <class kind="struct">singlepp::ClassifyIntegratedBuffers</class>
    <class kind="struct">singlepp::ClassifyIntegratedOptions</class>
    <class kind="struct">singlepp::ClassifyIntegratedResults</class>
    <class kind="struct">singlepp::ClassifySingleBuffers</class>
    <class kind="struct">singlepp::ClassifySingleOptions</class>
    <class kind="struct">singlepp::ClassifySingleResults</class>
    <class kind="class">singlepp::TrainedIntegrated</class>
    <class kind="class">singlepp::TrainedSingle</class>
    <class kind="class">singlepp::TrainedSingleIntersect</class>
    <class kind="struct">singlepp::TrainIntegratedInput</class>
    <class kind="struct">singlepp::TrainIntegratedOptions</class>
    <class kind="struct">singlepp::TrainSingleOptions</class>
    <member kind="typedef">
      <type>std::vector&lt; std::vector&lt; std::vector&lt; Index_ &gt; &gt; &gt;</type>
      <name>Markers</name>
      <anchorfile>namespacesinglepp.html</anchorfile>
      <anchor>a87e4be0939142d883ad98944e6336a5c</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>std::vector&lt; std::pair&lt; Index_, Index_ &gt; &gt;</type>
      <name>Intersection</name>
      <anchorfile>namespacesinglepp.html</anchorfile>
      <anchor>a24793af2399b65d4f290e28cc0ef475c</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>classify_single</name>
      <anchorfile>namespacesinglepp.html</anchorfile>
      <anchor>ae8772734badf4a09d6c285b9b98295ec</anchor>
      <arglist>(const tatami::Matrix&lt; Value_, Index_ &gt; &amp;test, const TrainedSingle&lt; Index_, Float_ &gt; &amp;trained, const ClassifySingleBuffers&lt; Label_, Float_ &gt; &amp;buffers, const ClassifySingleOptions&lt; Float_ &gt; &amp;options)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>classify_single_intersect</name>
      <anchorfile>namespacesinglepp.html</anchorfile>
      <anchor>ad7d214a5da53597d24dec5cec59e3beb</anchor>
      <arglist>(const tatami::Matrix&lt; Value_, Index_ &gt; &amp;test, const TrainedSingleIntersect&lt; Index_, Float_ &gt; &amp;trained, const ClassifySingleBuffers&lt; Label_, Float_ &gt; &amp;buffers, const ClassifySingleOptions&lt; Float_ &gt; &amp;options)</arglist>
    </member>
    <member kind="function">
      <type>ClassifySingleResults&lt; Label_, Float_ &gt;</type>
      <name>classify_single</name>
      <anchorfile>namespacesinglepp.html</anchorfile>
      <anchor>af23220206f86fe833bad096526d58db0</anchor>
      <arglist>(const tatami::Matrix&lt; Value_, Index_ &gt; &amp;test, const TrainedSingle&lt; Index_, Float_ &gt; &amp;trained, const ClassifySingleOptions&lt; Float_ &gt; &amp;options)</arglist>
    </member>
    <member kind="function">
      <type>ClassifySingleResults&lt; Label_, Float_ &gt;</type>
      <name>classify_single_intersect</name>
      <anchorfile>namespacesinglepp.html</anchorfile>
      <anchor>aaf9de543e62eb0ff85b7e5b701406e72</anchor>
      <arglist>(const tatami::Matrix&lt; Value_, Index_ &gt; &amp;test, const TrainedSingleIntersect&lt; Index_, Float_ &gt; &amp;trained, const ClassifySingleOptions&lt; Float_ &gt; &amp;options)</arglist>
    </member>
    <member kind="function">
      <type>TrainedSingle&lt; Index_, Float_ &gt;</type>
      <name>train_single</name>
      <anchorfile>namespacesinglepp.html</anchorfile>
      <anchor>abcea821671f9339e4e933be2ed123c6e</anchor>
      <arglist>(const tatami::Matrix&lt; Value_, Index_ &gt; &amp;ref, const Label_ *labels, Markers&lt; Index_ &gt; markers, const TrainSingleOptions&lt; Index_, Float_ &gt; &amp;options)</arglist>
    </member>
    <member kind="function">
      <type>TrainedSingleIntersect&lt; Index_, Float_ &gt;</type>
      <name>train_single_intersect</name>
      <anchorfile>namespacesinglepp.html</anchorfile>
      <anchor>a606d38abb5608efdfdb0c3bdb945ef23</anchor>
      <arglist>(const Intersection&lt; Index_ &gt; &amp;intersection, const tatami::Matrix&lt; Value_, Index_ &gt; &amp;ref, const Label_ *labels, Markers&lt; Index_ &gt; markers, const TrainSingleOptions&lt; Index_, Float_ &gt; &amp;options)</arglist>
    </member>
    <member kind="function">
      <type>TrainedSingleIntersect&lt; Index_, Float_ &gt;</type>
      <name>train_single_intersect</name>
      <anchorfile>namespacesinglepp.html</anchorfile>
      <anchor>a11341bf58d0086e0ec03a5bafb91bcf3</anchor>
      <arglist>(Index_ test_nrow, const Id_ *test_id, const tatami::Matrix&lt; Value_, Index_ &gt; &amp;ref, const Id_ *ref_id, const Label_ *labels, Markers&lt; Index_ &gt; markers, const TrainSingleOptions&lt; Index_, Float_ &gt; &amp;options)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>classify_integrated</name>
      <anchorfile>namespacesinglepp.html</anchorfile>
      <anchor>a60f705dc3fdc122139e2102713e7ec41</anchor>
      <arglist>(const tatami::Matrix&lt; Value_, Index_ &gt; &amp;test, const std::vector&lt; const Label_ * &gt; &amp;assigned, const TrainedIntegrated&lt; Index_ &gt; &amp;trained, ClassifyIntegratedBuffers&lt; RefLabel_, Float_ &gt; &amp;buffers, const ClassifyIntegratedOptions&lt; Float_ &gt; &amp;options)</arglist>
    </member>
    <member kind="function">
      <type>ClassifyIntegratedResults&lt; RefLabel_, Float_ &gt;</type>
      <name>classify_integrated</name>
      <anchorfile>namespacesinglepp.html</anchorfile>
      <anchor>aca35f5c98162213d14d4ce801b6376b0</anchor>
      <arglist>(const tatami::Matrix&lt; Value_, Index_ &gt; &amp;test, const std::vector&lt; const Label_ * &gt; &amp;assigned, const TrainedIntegrated&lt; Index_ &gt; &amp;trained, const ClassifyIntegratedOptions&lt; Float_ &gt; &amp;options)</arglist>
    </member>
    <member kind="function">
      <type>TrainIntegratedInput&lt; Value_, Index_, Label_ &gt;</type>
      <name>prepare_integrated_input</name>
      <anchorfile>namespacesinglepp.html</anchorfile>
      <anchor>a280f3bfb8b5a2e7e51233bc7382a8db1</anchor>
      <arglist>(const tatami::Matrix&lt; Value_, Index_ &gt; &amp;ref, const Label_ *labels, const TrainedSingle&lt; Index_, Float_ &gt; &amp;trained)</arglist>
    </member>
    <member kind="function">
      <type>TrainIntegratedInput&lt; Value_, Index_, Label_ &gt;</type>
      <name>prepare_integrated_input_intersect</name>
      <anchorfile>namespacesinglepp.html</anchorfile>
      <anchor>a00d24ce81780d784bf9809afdad00736</anchor>
      <arglist>(const Intersection&lt; Index_ &gt; &amp;intersection, const tatami::Matrix&lt; Value_, Index_ &gt; &amp;ref, const Label_ *labels, const TrainedSingleIntersect&lt; Index_, Float_ &gt; &amp;trained)</arglist>
    </member>
    <member kind="function">
      <type>TrainIntegratedInput&lt; Value_, Index_, Label_ &gt;</type>
      <name>prepare_integrated_input_intersect</name>
      <anchorfile>namespacesinglepp.html</anchorfile>
      <anchor>a55e0f072fe5fa1844313198ef1e64918</anchor>
      <arglist>(Index_ test_nrow, const Id_ *test_id, const tatami::Matrix&lt; Value_, Index_ &gt; &amp;ref, const Id_ *ref_id, const Label_ *labels, const TrainedSingleIntersect&lt; Index_, Float_ &gt; &amp;trained)</arglist>
    </member>
    <member kind="function">
      <type>TrainedIntegrated&lt; Index_ &gt;</type>
      <name>train_integrated</name>
      <anchorfile>namespacesinglepp.html</anchorfile>
      <anchor>af198ad22e020701691c9bb8326401a46</anchor>
      <arglist>(const std::vector&lt; TrainIntegratedInput&lt; Value_, Index_, Label_ &gt; &gt; &amp;inputs, const TrainIntegratedOptions &amp;options)</arglist>
    </member>
    <member kind="function">
      <type>TrainedIntegrated&lt; Index_ &gt;</type>
      <name>train_integrated</name>
      <anchorfile>namespacesinglepp.html</anchorfile>
      <anchor>a82e0dfffaed84685432ad2a39cc12a97</anchor>
      <arglist>(std::vector&lt; TrainIntegratedInput&lt; Value_, Index_, Label_ &gt; &gt; &amp;&amp;inputs, const TrainIntegratedOptions &amp;options)</arglist>
    </member>
    <member kind="function">
      <type>Intersection&lt; Index_ &gt;</type>
      <name>intersect_genes</name>
      <anchorfile>namespacesinglepp.html</anchorfile>
      <anchor>a22b70c4d5949e28af164723543361214</anchor>
      <arglist>(Index_ test_nrow, const Id_ *test_id, Index_ ref_nrow, const Id_ *ref_id)</arglist>
    </member>
    <member kind="function">
      <type>size_t</type>
      <name>number_of_classic_markers</name>
      <anchorfile>namespacesinglepp.html</anchorfile>
      <anchor>ac26ab7d42558905cab72d85e509cfb0c</anchor>
      <arglist>(size_t num_labels)</arglist>
    </member>
    <member kind="function">
      <type>Markers&lt; Index_ &gt;</type>
      <name>choose_classic_markers</name>
      <anchorfile>namespacesinglepp.html</anchorfile>
      <anchor>ae888912441a04acf856e74a9f4a0452b</anchor>
      <arglist>(const std::vector&lt; const tatami::Matrix&lt; Value_, Index_ &gt; * &gt; &amp;representatives, const std::vector&lt; const Label_ * &gt; &amp;labels, const ChooseClassicMarkersOptions &amp;options)</arglist>
    </member>
    <member kind="function">
      <type>Markers&lt; Index_ &gt;</type>
      <name>choose_classic_markers</name>
      <anchorfile>namespacesinglepp.html</anchorfile>
      <anchor>ae6b37c9c0ac535f7c85771fbeb98e4df</anchor>
      <arglist>(const tatami::Matrix&lt; Value_, Index_ &gt; &amp;representative, const Label_ *labels, const ChooseClassicMarkersOptions &amp;options)</arglist>
    </member>
  </compound>
  <compound kind="page">
    <name>index</name>
    <title>C++ port of SingleR</title>
    <filename>index.html</filename>
    <docanchor file="index.html">md__2github_2workspace_2README</docanchor>
  </compound>
</tagfile>
