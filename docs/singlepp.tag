<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>
<tagfile doxygen_version="1.12.0">
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
    <name>defs.hpp</name>
    <path>singlepp/</path>
    <filename>defs_8hpp.html</filename>
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
      <type>Float_</type>
      <name>fine_tune_threshold</name>
      <anchorfile>structsinglepp_1_1ClassifyIntegratedOptions.html</anchorfile>
      <anchor>ae963907ed5ed64dd7c198d6171e54003</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>fine_tune</name>
      <anchorfile>structsinglepp_1_1ClassifyIntegratedOptions.html</anchorfile>
      <anchor>a4b00e9fdb3d16f431709932dd4dddca9</anchor>
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
      <type>std::size_t</type>
      <name>num_references</name>
      <anchorfile>classsinglepp_1_1TrainedIntegrated.html</anchorfile>
      <anchor>a5a0e980d87d8d47e2a319558b11aeec7</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>std::size_t</type>
      <name>num_labels</name>
      <anchorfile>classsinglepp_1_1TrainedIntegrated.html</anchorfile>
      <anchor>af7e28abc4fc7c9b1f2645af46db94e8c</anchor>
      <arglist>(std::size_t r) const</arglist>
    </member>
    <member kind="function">
      <type>std::size_t</type>
      <name>num_profiles</name>
      <anchorfile>classsinglepp_1_1TrainedIntegrated.html</anchorfile>
      <anchor>adf3bc8f97fc1d012dae31e24b3c5d19a</anchor>
      <arglist>(std::size_t r) const</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>singlepp::TrainedSingle</name>
    <filename>classsinglepp_1_1TrainedSingle.html</filename>
    <templarg>typename Index_</templarg>
    <templarg>typename Float_</templarg>
    <member kind="function">
      <type>Index_</type>
      <name>get_test_nrow</name>
      <anchorfile>classsinglepp_1_1TrainedSingle.html</anchorfile>
      <anchor>a7f4f34d6d10ebf69fc22b69cf8306666</anchor>
      <arglist>() const</arglist>
    </member>
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
      <type>std::size_t</type>
      <name>num_labels</name>
      <anchorfile>classsinglepp_1_1TrainedSingle.html</anchorfile>
      <anchor>acd1d604b21e8deb287ccd5210b431bf8</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>std::size_t</type>
      <name>num_profiles</name>
      <anchorfile>classsinglepp_1_1TrainedSingle.html</anchorfile>
      <anchor>ae1a317b134a501627abacd439f3425bd</anchor>
      <arglist>() const</arglist>
    </member>
  </compound>
  <compound kind="class">
    <name>singlepp::TrainedSingleIntersect</name>
    <filename>classsinglepp_1_1TrainedSingleIntersect.html</filename>
    <templarg>typename Index_</templarg>
    <templarg>typename Float_</templarg>
    <member kind="function">
      <type>Index_</type>
      <name>get_test_nrow</name>
      <anchorfile>classsinglepp_1_1TrainedSingleIntersect.html</anchorfile>
      <anchor>a5f836d3681b61bb1c5b6b9fc7fbb074f</anchor>
      <arglist>() const</arglist>
    </member>
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
      <type>std::size_t</type>
      <name>num_labels</name>
      <anchorfile>classsinglepp_1_1TrainedSingleIntersect.html</anchorfile>
      <anchor>ab18d9f51ddc258439aecbeea4bc758c0</anchor>
      <arglist>() const</arglist>
    </member>
    <member kind="function">
      <type>std::size_t</type>
      <name>num_profiles</name>
      <anchorfile>classsinglepp_1_1TrainedSingleIntersect.html</anchorfile>
      <anchor>a082bf3ff596b4f374933dc57fcc67f11</anchor>
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
    <member kind="variable">
      <type>int</type>
      <name>top</name>
      <anchorfile>structsinglepp_1_1TrainSingleOptions.html</anchorfile>
      <anchor>a8cf88a1fd176fb330c3d3a57cccda477</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>int</type>
      <name>num_threads</name>
      <anchorfile>structsinglepp_1_1TrainSingleOptions.html</anchorfile>
      <anchor>a76c303868ef63d76fd157ad80974d3a6</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="namespace">
    <name>singlepp</name>
    <filename>namespacesinglepp.html</filename>
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
      <anchor>a1d147dc88bf87bef188bd24b56d0f571</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>std::vector&lt; std::pair&lt; Index_, Index_ &gt; &gt;</type>
      <name>Intersection</name>
      <anchorfile>namespacesinglepp.html</anchorfile>
      <anchor>ae0c63a73c7c40ad5e6f47ef0f9abac5d</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>int</type>
      <name>DefaultIndex</name>
      <anchorfile>namespacesinglepp.html</anchorfile>
      <anchor>a75885e26a4870572229427b4acc3e74f</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>int</type>
      <name>DefaultLabel</name>
      <anchorfile>namespacesinglepp.html</anchorfile>
      <anchor>a16859be67879a6e0650877f12edd9f14</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>int</type>
      <name>DefaultRefLabel</name>
      <anchorfile>namespacesinglepp.html</anchorfile>
      <anchor>a9e8cb5a049cb39b52068562ef2c9bbff</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>double</type>
      <name>DefaultFloat</name>
      <anchorfile>namespacesinglepp.html</anchorfile>
      <anchor>a337b47b25c3bfaff91d30f38fe5bb31b</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>double</type>
      <name>DefaultValue</name>
      <anchorfile>namespacesinglepp.html</anchorfile>
      <anchor>a1ced86db19f4d2cf55ba4310bd2c16d2</anchor>
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
      <anchor>a5f8f8106b91cfe79435b20c9b38e500c</anchor>
      <arglist>(const tatami::Matrix&lt; Value_, Index_ &gt; &amp;ref, const Label_ *labels, Markers&lt; Index_ &gt; markers, const TrainSingleOptions &amp;options)</arglist>
    </member>
    <member kind="function">
      <type>TrainedSingleIntersect&lt; Index_, Float_ &gt;</type>
      <name>train_single_intersect</name>
      <anchorfile>namespacesinglepp.html</anchorfile>
      <anchor>ae9aef6aa0694cdeef9d111f8a66f8b05</anchor>
      <arglist>(Index_ test_nrow, const Intersection&lt; Index_ &gt; &amp;intersection, const tatami::Matrix&lt; Value_, Index_ &gt; &amp;ref, const Label_ *labels, Markers&lt; Index_ &gt; markers, const TrainSingleOptions &amp;options)</arglist>
    </member>
    <member kind="function">
      <type>TrainedSingleIntersect&lt; Index_, Float_ &gt;</type>
      <name>train_single_intersect</name>
      <anchorfile>namespacesinglepp.html</anchorfile>
      <anchor>a35750e91dab76c79356720580c10c875</anchor>
      <arglist>(Index_ test_nrow, const Id_ *test_id, const tatami::Matrix&lt; Value_, Index_ &gt; &amp;ref, const Id_ *ref_id, const Label_ *labels, Markers&lt; Index_ &gt; markers, const TrainSingleOptions &amp;options)</arglist>
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
      <anchor>a5da95a907514caf12c99a9f3fd83aa20</anchor>
      <arglist>(Index_ test_nrow, const Intersection&lt; Index_ &gt; &amp;intersection, const tatami::Matrix&lt; Value_, Index_ &gt; &amp;ref, const Label_ *labels, const TrainedSingleIntersect&lt; Index_, Float_ &gt; &amp;trained)</arglist>
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
  </compound>
  <compound kind="page">
    <name>index</name>
    <title>C++ port of SingleR</title>
    <filename>index.html</filename>
    <docanchor file="index.html">md__2github_2workspace_2README</docanchor>
  </compound>
</tagfile>
